"""Turn free-text training notes into structured ``Workout`` records.

Two extractors share one interface:

* ``LLMExtractor`` forces Claude to call a ``record_workout`` tool whose input schema is the
  ``Workout`` JSON schema, then validates the result with Pydantic. Invalid output is never
  stored: validation errors are raised to the caller.
* ``RuleExtractor`` is a regex baseline. It runs offline and gives the eval harness a
  reference point, so improvements from the LLM are measured rather than assumed.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Protocol

from .llm import LLMClient
from .schemas import Feeling, Workout, WorkoutType

SYSTEM_PROMPT = """You convert a runner's free-text training note into structured data.
Rules:
- Only record values that are stated or directly computable from the note. Never guess.
- Convert miles to km (1 mile = 1.609 km) and paces to decimal minutes per km.
- If distance and duration are given but pace is not, compute pace = duration / distance.
- Symptoms are short lowercase phrases (e.g. "heavy legs", "calf tightness").
- The summary is one neutral sentence.
Always answer by calling the record_workout tool."""

TOOL_NAME = "record_workout"


def workout_tool() -> dict[str, object]:
    return {
        "name": TOOL_NAME,
        "description": "Record one structured training session extracted from the note.",
        "input_schema": Workout.model_json_schema(),
    }


class Extractor(Protocol):
    def extract(self, note: str, workout_date: date | None = None) -> Workout: ...


class LLMExtractor:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def extract(self, note: str, workout_date: date | None = None) -> Workout:
        context = f"Session date: {workout_date.isoformat()}\n" if workout_date else ""
        messages = [{"role": "user", "content": f"{context}Note:\n{note}"}]

        def ask(tool_choice: dict[str, str]) -> Any:
            return self._llm.create(
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=[workout_tool()],
                tool_choice=tool_choice,
                max_tokens=800,
            )

        try:
            # Strict mode: the model must answer by calling record_workout.
            response = ask({"type": "tool", "name": TOOL_NAME})
        except Exception as exc:
            # Some models (e.g. with extended thinking) reject forced tool choice.
            # Fall back to letting the model choose; the prompt still requires the tool,
            # and the checks below still reject anything that isn't a valid tool call.
            if "tool_choice" not in str(exc):
                raise
            response = ask({"type": "auto"})

        for block in response.content:
            if block.type == "tool_use" and block.name == TOOL_NAME:
                workout = Workout.model_validate(block.input)
                if workout_date and workout.workout_date is None:
                    workout = workout.model_copy(update={"workout_date": workout_date})
                return workout
        raise ValueError("Model did not call the record_workout tool")


_TYPE_KEYWORDS: list[tuple[WorkoutType, tuple[str, ...]]] = [
    (WorkoutType.REST, ("rest day", "day off", "no run")),
    (WorkoutType.RACE, ("race", "marathon", "half marathon", "parkrun", "pb")),
    (WorkoutType.INTERVALS, ("interval", "repeats", "x 400", "x400", "x 800", "x800", "x 1k", "track")),
    (WorkoutType.TEMPO, ("tempo", "threshold", "lactate")),
    (WorkoutType.LONG, ("long run", "long")),
    (WorkoutType.RECOVERY, ("recovery", "shakeout", "shake-out")),
    (WorkoutType.EASY, ("easy", "aerobic", "zone 2", "z2")),
]

_SYMPTOMS = (
    "heavy legs", "tired legs", "calf tightness", "tight calf", "sore knee", "knee pain",
    "shin pain", "hamstring tightness", "side stitch", "fatigue", "dizzy", "cramp",
)
_MILES_TO_KM = 1.609


class RuleExtractor:
    """Regex baseline extractor; deliberately simple and transparent."""

    def extract(self, note: str, workout_date: date | None = None) -> Workout:
        text = note.lower()
        wtype = next(
            (t for t, words in _TYPE_KEYWORDS if any(w in text for w in words)), WorkoutType.OTHER
        )

        distance = None
        if m := re.search(r"(\d+(?:\.\d+)?)\s*(km|k|mi|miles?)\b", text):
            value = float(m.group(1))
            distance = round(value * _MILES_TO_KM, 2) if m.group(2).startswith("mi") else value

        duration = None
        if m := re.search(r"\b(\d{1,2}):(\d{2}):(\d{2})\b", text):
            duration = int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 60
        elif m := re.search(r"(\d+(?:\.\d+)?)\s*(?:min|mins|minutes)\b", text):
            duration = float(m.group(1))

        pace = None
        if m := re.search(r"(\d{1,2}):(\d{2})\s*(?:/km|per km|min/km)", text):
            pace = int(m.group(1)) + int(m.group(2)) / 60
        elif distance and duration:
            pace = duration / distance

        hr = None
        if m := re.search(r"(?:avg\s*)?(?:hr|heart rate)\s*(?:of|was|at|:)?\s*(\d{2,3})", text):
            hr = int(m.group(1))

        rpe = None
        if m := re.search(r"rpe\s*(?:of|:)?\s*(\d{1,2})", text):
            rpe = min(10, max(1, int(m.group(1))))

        feeling = None
        for word, mood in (("great", Feeling.GREAT), ("felt good", Feeling.GOOD),
                           ("terrible", Feeling.BAD), ("awful", Feeling.BAD),
                           ("felt bad", Feeling.BAD), ("ok", Feeling.OK)):
            if word in text:
                feeling = mood
                break

        symptoms = [s for s in _SYMPTOMS if s in text]
        summary = f"{wtype.value.capitalize()} session" + (f" of {distance:g} km." if distance else ".")
        return Workout(
            workout_date=workout_date,
            workout_type=wtype,
            distance_km=distance,
            duration_min=round(duration, 2) if duration else None,
            avg_pace_min_per_km=round(pace, 2) if pace else None,
            avg_hr=hr,
            rpe=rpe,
            feeling=feeling,
            symptoms=symptoms,
            summary=summary,
        )