from __future__ import annotations

import json
from datetime import date

from fastapi.testclient import TestClient

from runcoach import analytics
from runcoach.agent import CoachAgent
from runcoach.api import create_app
from runcoach.config import Settings
from runcoach.extract import TOOL_NAME
from runcoach.garmin import import_csv
from runcoach.llm import Block, Response, ScriptedLLM

TODAY = date(2026, 10, 25)


def test_weekly_mileage_ignores_non_running(session, sample_csv: str) -> None:
    import_csv(session, sample_csv)
    weeks = analytics.weekly_mileage(session, weeks=8, today=TODAY)["weeks"]
    assert len(weeks) == 8 and all(w["runs"] <= 5 for w in weeks)
    assert sum(w["km"] for w in weeks) > 300


def test_agent_executes_tools_and_returns_answer(session, sample_csv: str) -> None:
    import_csv(session, sample_csv)
    llm = ScriptedLLM([
        Response([Block(type="tool_use", id="c1", name="get_weekly_mileage", input={"weeks": 4})], "tool_use"),
        Response([Block(type="text", text="Your mileage is building steadily.")], "end_turn"),
    ])
    result = CoachAgent(llm).ask(session, "How is my training going?")
    assert result.answer == "Your mileage is building steadily."
    assert result.tools_used == ["get_weekly_mileage"]
    tool_result = llm.requests[1]["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result" and not tool_result["is_error"]
    assert len(json.loads(tool_result["content"])["weeks"]) == 4


def test_agent_reports_unknown_tool_as_error(session) -> None:
    llm = ScriptedLLM([
        Response([Block(type="tool_use", id="c1", name="drop_tables", input={})], "tool_use"),
        Response([Block(type="text", text="I can't do that.")], "end_turn"),
    ])
    result = CoachAgent(llm).ask(session, "delete everything")
    assert llm.requests[1]["messages"][-1]["content"][0]["is_error"] is True
    assert result.answer == "I can't do that."


def test_agent_stops_after_max_steps(session) -> None:
    loop = [Response([Block(type="tool_use", id=f"c{i}", name="get_recent_activities", input={})], "tool_use")
            for i in range(3)]
    result = CoachAgent(ScriptedLLM(loop), max_steps=3).ask(session, "?")
    assert result.answer.startswith("Stopped") and len(result.tools_used) == 3


def _settings() -> Settings:
    return Settings(database_url="sqlite:///:memory:", anthropic_api_key=None, model="test", max_agent_steps=4)


def test_api_end_to_end_with_scripted_llm(session_factory, sample_csv: str) -> None:
    payload = {"workout_type": "easy", "distance_km": 10, "summary": "Easy 10 km."}
    llm = ScriptedLLM([
        Response([Block(type="tool_use", id="t1", name=TOOL_NAME, input=payload)], "tool_use"),
        Response([Block(type="text", text="Looks good.")], "end_turn"),
    ])
    client = TestClient(create_app(_settings(), llm=llm, session_factory=session_factory))

    assert client.get("/health").json()["extractor"] == "llm"
    r = client.post("/import/garmin", files={"file": ("a.csv", sample_csv.encode(), "text/csv")})
    assert r.status_code == 200 and r.json()["inserted"] > 0
    r = client.post("/notes", json={"text": "easy 10k", "workout_date": "2026-10-20"})
    assert r.status_code == 200 and r.json()["workout_date"] == "2026-10-20"
    r = client.post("/ask", json={"question": "How am I doing?"})
    assert r.json() == {"answer": "Looks good.", "tools_used": []}


def test_api_without_key_uses_rule_baseline(session_factory) -> None:
    client = TestClient(create_app(_settings(), llm=None, session_factory=session_factory))
    assert client.get("/health").json()["extractor"] == "rules"
    assert client.post("/notes", json={"text": "Tempo 8 km in 36:00"}).json()["workout_type"] == "tempo"
    assert client.post("/ask", json={"question": "hi"}).status_code == 503
