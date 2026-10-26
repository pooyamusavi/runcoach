"""Coaching agent: Claude answers training questions by calling analytics tools.

The loop is explicit and bounded: the model may request tools, we execute them against the
database, return the results, and stop after ``max_steps`` round trips. Every tool call is
recorded so the API can show which data an answer was based on.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from . import analytics
from .llm import LLMClient

SYSTEM_PROMPT = """You are an evidence-based running coach.
Answer the athlete's question using the tools to look at their actual training data.
- Base every claim on tool results; if the data is missing, say so.
- Be concise: a short answer, then 2-4 specific, actionable suggestions.
- Flag warning signs (sudden mileage jumps over ~10%/week, recurring pain symptoms) and
  recommend seeing a professional for injuries. You are not a medical provider."""

ToolFn = Callable[[Session, dict[str, Any]], dict[str, Any]]


def _int_arg(args: dict[str, Any], key: str, default: int, low: int, high: int) -> int:
    try:
        value = int(args.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


TOOLS: dict[str, tuple[dict[str, Any], ToolFn]] = {
    "get_weekly_mileage": (
        {
            "name": "get_weekly_mileage",
            "description": "Weekly running distance, number of runs and hours for the last N weeks.",
            "input_schema": {
                "type": "object",
                "properties": {"weeks": {"type": "integer", "minimum": 1, "maximum": 52}},
            },
        },
        lambda s, a: analytics.weekly_mileage(s, _int_arg(a, "weeks", 8, 1, 52)),
    ),
    "get_long_run_trend": (
        {
            "name": "get_long_run_trend",
            "description": "Longest run of each week with pace and average heart rate.",
            "input_schema": {
                "type": "object",
                "properties": {"weeks": {"type": "integer", "minimum": 1, "maximum": 52}},
            },
        },
        lambda s, a: analytics.long_run_trend(s, _int_arg(a, "weeks", 8, 1, 52)),
    ),
    "get_recent_activities": (
        {
            "name": "get_recent_activities",
            "description": "Most recent activities with distance, duration, pace and heart rate.",
            "input_schema": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 50}},
            },
        },
        lambda s, a: analytics.recent_activities(s, _int_arg(a, "limit", 10, 1, 50)),
    ),
    "get_training_notes": (
        {
            "name": "get_training_notes",
            "description": "Recent free-text training notes and their structured extraction "
            "(feeling, RPE, symptoms).",
            "input_schema": {
                "type": "object",
                "properties": {"days": {"type": "integer", "minimum": 1, "maximum": 90}},
            },
        },
        lambda s, a: analytics.recent_notes(s, _int_arg(a, "days", 14, 1, 90)),
    ),
}


@dataclass
class AgentResult:
    answer: str
    tools_used: list[str] = field(default_factory=list)


class CoachAgent:
    def __init__(self, llm: LLMClient, max_steps: int = 6) -> None:
        self._llm = llm
        self._max_steps = max_steps

    def ask(self, session: Session, question: str) -> AgentResult:
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        tool_specs = [spec for spec, _ in TOOLS.values()]
        used: list[str] = []

        for _ in range(self._max_steps):
            response = self._llm.create(
                system=SYSTEM_PROMPT, messages=messages, tools=tool_specs, max_tokens=1200
            )
            tool_calls = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_calls:
                text = "".join(b.text for b in response.content if b.type == "text").strip()
                return AgentResult(answer=text or "I could not produce an answer.", tools_used=used)

            messages.append({"role": "assistant", "content": [_block_to_dict(b) for b in response.content]})
            results = []
            for call in tool_calls:
                used.append(call.name)
                if call.name in TOOLS:
                    try:
                        payload = TOOLS[call.name][1](session, dict(call.input or {}))
                        content, is_error = json.dumps(payload, default=str), False
                    except Exception as exc:  # report tool failures to the model, don't crash
                        content, is_error = f"Tool error: {exc}", True
                else:
                    content, is_error = f"Unknown tool: {call.name}", True
                results.append(
                    {"type": "tool_result", "tool_use_id": call.id, "content": content, "is_error": is_error}
                )
            messages.append({"role": "user", "content": results})

        return AgentResult(answer="Stopped after the maximum number of tool steps.", tools_used=used)


def _block_to_dict(block: Any) -> dict[str, Any]:
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": dict(block.input or {})}
    return {"type": block.type}
