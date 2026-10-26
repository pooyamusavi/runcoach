"""Import Garmin Connect "Activities.csv" exports.

The import is idempotent: each row is upserted on its natural key (start time + activity
type), so re-importing an overlapping export updates rows instead of duplicating them.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import Activity

_MISSING = {"", "--", "-"}


def _num(value: str | None) -> float | None:
    if value is None or value.strip() in _MISSING:
        return None
    return float(value.replace(",", ""))


def _int(value: str | None) -> int | None:
    number = _num(value)
    return int(round(number)) if number is not None else None


def _minutes(value: str | None) -> float | None:
    """Parse 'HH:MM:SS', 'MM:SS' or 'HH:MM:SS.s' into decimal minutes."""
    if value is None or value.strip() in _MISSING:
        return None
    parts = [float(p) for p in value.strip().split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, (minutes, seconds) = 0.0, parts
    else:
        raise ValueError(f"Unrecognised time value: {value!r}")
    return round(hours * 60 + minutes + seconds / 60, 2)


def _start_time(value: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%m/%d/%Y %H:%M"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date value: {value!r}")


@dataclass
class ImportResult:
    inserted: int = 0
    updated: int = 0
    skipped: int = 0


def parse_rows(csv_text: str) -> list[dict[str, object]]:
    reader = csv.DictReader(io.StringIO(csv_text))
    rows: list[dict[str, object]] = []
    for raw in reader:
        if not raw.get("Date"):
            continue
        rows.append(
            {
                "start_time": _start_time(raw["Date"]),
                "activity_type": (raw.get("Activity Type") or "Unknown").strip(),
                "title": (raw.get("Title") or "").strip(),
                "distance_km": _num(raw.get("Distance")),
                "duration_min": _minutes(raw.get("Time")),
                "avg_hr": _int(raw.get("Avg HR")),
                "max_hr": _int(raw.get("Max HR")),
                "avg_pace_min_per_km": _minutes(raw.get("Avg Pace")),
                "total_ascent_m": _num(raw.get("Total Ascent")),
            }
        )
    return rows


def import_csv(session: Session, csv_text: str) -> ImportResult:
    result = ImportResult()
    for row in parse_rows(csv_text):
        existing = session.scalar(
            select(Activity).where(
                Activity.start_time == row["start_time"],
                Activity.activity_type == row["activity_type"],
            )
        )
        if existing is None:
            session.add(Activity(**row))
            result.inserted += 1
            continue
        changed = False
        for key, value in row.items():
            if getattr(existing, key) != value:
                setattr(existing, key, value)
                changed = True
        if changed:
            result.updated += 1
        else:
            result.skipped += 1
    session.flush()
    return result
