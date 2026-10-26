"""Typed schemas shared by the extractor, database layer and API."""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class WorkoutType(str, Enum):
    EASY = "easy"
    RECOVERY = "recovery"
    LONG = "long"
    TEMPO = "tempo"
    INTERVALS = "intervals"
    RACE = "race"
    REST = "rest"
    OTHER = "other"


class Feeling(str, Enum):
    GREAT = "great"
    GOOD = "good"
    OK = "ok"
    BAD = "bad"


class Workout(BaseModel):
    """Structured representation of one free-text training note."""

    workout_date: date | None = Field(
        default=None, description="Date of the session if stated or inferable, ISO format."
    )
    workout_type: WorkoutType = Field(description="Kind of session.")
    distance_km: float | None = Field(
        default=None, ge=0, description="Distance in kilometres. Convert miles (1 mi = 1.609 km)."
    )
    duration_min: float | None = Field(default=None, ge=0, description="Moving time in minutes.")
    avg_pace_min_per_km: float | None = Field(
        default=None, ge=0, description="Average pace in decimal minutes per km (5:30/km = 5.5)."
    )
    avg_hr: int | None = Field(default=None, ge=30, le=230, description="Average heart rate, bpm.")
    rpe: int | None = Field(
        default=None, ge=1, le=10, description="Rate of perceived exertion, 1-10, only if stated."
    )
    feeling: Feeling | None = Field(default=None, description="Overall feeling, only if expressed.")
    symptoms: list[str] = Field(
        default_factory=list,
        description="Short lowercase physical or mental symptoms, e.g. 'heavy legs', 'calf tightness'.",
    )
    summary: str = Field(description="One-sentence neutral summary of the session.")


class NoteIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    workout_date: date | None = None


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class AskOut(BaseModel):
    answer: str
    tools_used: list[str]
