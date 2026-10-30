"""Training analytics computed from stored activities and notes.

These functions are the tools the coaching agent can call, so they return plain,
JSON-serialisable dictionaries.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import Activity, TrainingNote

RUNNING_TYPES = ("Running", "Treadmill Running", "Trail Running", "Track Running")


def _runs_since(session: Session, since: datetime) -> list[Activity]:
    stmt = (
        select(Activity)
        .where(Activity.start_time >= since, Activity.activity_type.in_(RUNNING_TYPES))
        .order_by(Activity.start_time)
    )
    return list(session.scalars(stmt))


def _week_start(day: date) -> date:
    return day - timedelta(days=day.weekday())


def weekly_mileage(session: Session, weeks: int = 8, today: date | None = None) -> dict[str, Any]:
    today = today or date.today()
    start = _week_start(today) - timedelta(weeks=weeks - 1)
    totals: dict[date, dict[str, float]] = defaultdict(lambda: {"km": 0.0, "runs": 0, "minutes": 0.0})
    for run in _runs_since(session, datetime.combine(start, datetime.min.time())):
        week = _week_start(run.start_time.date())
        totals[week]["km"] += run.distance_km or 0.0
        totals[week]["minutes"] += run.duration_min or 0.0
        totals[week]["runs"] += 1
    rows = []
    for i in range(weeks):
        week = start + timedelta(weeks=i)
        t = totals.get(week, {"km": 0.0, "runs": 0, "minutes": 0.0})
        rows.append({"week_start": week.isoformat(), "km": round(t["km"], 1),
                     "runs": int(t["runs"]), "hours": round(t["minutes"] / 60, 1)})
    return {"weeks": rows}


def long_run_trend(session: Session, weeks: int = 8, today: date | None = None) -> dict[str, Any]:
    """Longest run per week with its pace and heart rate, to spot aerobic progress or drift."""
    today = today or date.today()
    start = _week_start(today) - timedelta(weeks=weeks - 1)
    longest: dict[date, Activity] = {}
    for run in _runs_since(session, datetime.combine(start, datetime.min.time())):
        week = _week_start(run.start_time.date())
        if run.distance_km and (week not in longest or run.distance_km > (longest[week].distance_km or 0)):
            longest[week] = run
    return {
        "long_runs": [
            {
                "week_start": week.isoformat(),
                "date": run.start_time.date().isoformat(),
                "km": run.distance_km,
                "pace_min_per_km": run.avg_pace_min_per_km,
                "avg_hr": run.avg_hr,
            }
            for week, run in sorted(longest.items())
        ]
    }


def training_load(session: Session, today: date | None = None) -> dict[str, Any]:
    """Acute:chronic workload ratio: last 7 days of running vs. the average weekly
    distance of the last 28 days. A ratio above 1.3 flags a sharp, injury-risking jump
    in load; ``ratio`` is ``None`` when there's no chronic data to compare against."""
    today = today or date.today()
    acute_start = today - timedelta(days=6)
    chronic_start = today - timedelta(days=27)
    runs = _runs_since(session, datetime.combine(chronic_start, datetime.min.time()))
    acute_km = sum(r.distance_km or 0.0 for r in runs
                   if acute_start <= r.start_time.date() <= today)
    chronic_km_total = sum(r.distance_km or 0.0 for r in runs if r.start_time.date() <= today)
    chronic_km_per_week = chronic_km_total / 4
    ratio = round(acute_km / chronic_km_per_week, 2) if chronic_km_per_week > 0 else None
    return {
        "acute_km": round(acute_km, 1),
        "chronic_km_per_week": round(chronic_km_per_week, 1),
        "ratio": ratio,
        "warning": ratio is not None and ratio > 1.3,
    }


def recent_activities(session: Session, limit: int = 10) -> dict[str, Any]:
    stmt = select(Activity).order_by(Activity.start_time.desc()).limit(min(limit, 50))
    return {
        "activities": [
            {
                "date": a.start_time.isoformat(timespec="minutes"),
                "type": a.activity_type,
                "title": a.title,
                "km": a.distance_km,
                "minutes": a.duration_min,
                "pace_min_per_km": a.avg_pace_min_per_km,
                "avg_hr": a.avg_hr,
            }
            for a in session.scalars(stmt)
        ]
    }


def recent_notes(session: Session, days: int = 14, today: date | None = None) -> dict[str, Any]:
    today = today or date.today()
    since = today - timedelta(days=days)
    stmt = (
        select(TrainingNote)
        .where((TrainingNote.workout_date >= since) | (TrainingNote.workout_date.is_(None)))
        .order_by(TrainingNote.id.desc())
        .limit(30)
    )
    return {
        "notes": [
            {"date": n.workout_date.isoformat() if n.workout_date else None,
             "text": n.raw_text, "extracted": n.extracted}
            for n in session.scalars(stmt)
        ]
    }
