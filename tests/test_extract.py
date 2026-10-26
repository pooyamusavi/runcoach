from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from runcoach.extract import TOOL_NAME, LLMExtractor, RuleExtractor
from runcoach.llm import Block, Response, ScriptedLLM
from runcoach.schemas import WorkoutType


def test_rule_extractor_basic_fields() -> None:
    w = RuleExtractor().extract("Easy 10k this morning, 55 min, HR 138. Heavy legs.")
    assert w.workout_type is WorkoutType.EASY
    assert w.distance_km == 10.0 and w.duration_min == 55.0
    assert w.avg_pace_min_per_km == pytest.approx(5.5)
    assert w.avg_hr == 138 and "heavy legs" in w.symptoms


def test_rule_extractor_converts_miles() -> None:
    w = RuleExtractor().extract("6 miles recovery, 62 minutes")
    assert w.distance_km == pytest.approx(9.65, abs=0.01)


def test_llm_extractor_forces_tool_and_validates() -> None:
    payload = {"workout_type": "long", "distance_km": 28, "avg_pace_min_per_km": 5.67,
               "symptoms": ["calf tightness"], "summary": "Long run of 28 km."}
    llm = ScriptedLLM([Response([Block(type="tool_use", id="t1", name=TOOL_NAME, input=payload)], "tool_use")])
    w = LLMExtractor(llm).extract("Long run 28 km...", workout_date=date(2026, 10, 4))
    assert w.workout_type is WorkoutType.LONG and w.workout_date == date(2026, 10, 4)
    assert llm.requests[0]["tools"][0]["name"] == TOOL_NAME


def test_llm_extractor_rejects_invalid_output() -> None:
    bad = {"workout_type": "long", "avg_hr": 400, "summary": "x"}  # impossible heart rate
    llm = ScriptedLLM([Response([Block(type="tool_use", id="t1", name=TOOL_NAME, input=bad)], "tool_use")])
    with pytest.raises(ValidationError):
        LLMExtractor(llm).extract("...")


def test_llm_extractor_requires_tool_call() -> None:
    llm = ScriptedLLM([Response([Block(type="text", text="Sure!")], "end_turn")])
    with pytest.raises(ValueError):
        LLMExtractor(llm).extract("...")
