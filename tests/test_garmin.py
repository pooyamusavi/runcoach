from __future__ import annotations

from sqlalchemy import func, select

from runcoach.db import Activity
from runcoach.garmin import _minutes, import_csv, parse_rows


def test_parse_time_formats() -> None:
    assert _minutes("00:55:00") == 55.0
    assert _minutes("5:30") == 5.5
    assert _minutes("1:39:30") == 99.5
    assert _minutes("--") is None


def test_parse_rows_handles_missing_values(sample_csv: str) -> None:
    rows = parse_rows(sample_csv)
    strength = [r for r in rows if r["activity_type"] == "Strength Training"]
    assert strength and strength[0]["distance_km"] is None and strength[0]["avg_pace_min_per_km"] is None


def test_import_is_idempotent(session, sample_csv: str) -> None:
    first = import_csv(session, sample_csv)
    second = import_csv(session, sample_csv)
    total = session.scalar(select(func.count()).select_from(Activity))
    assert first.inserted == total > 0
    assert second.inserted == 0 and second.updated == 0 and second.skipped == total


def test_reimport_updates_changed_rows(session, sample_csv: str) -> None:
    import_csv(session, sample_csv)
    header, first_row, *rest = sample_csv.splitlines()
    cols = first_row.split(",")
    cols[header.split(",").index("Avg HR")] = "199"
    result = import_csv(session, "\n".join([header, ",".join(cols)]))
    assert result.updated == 1 and result.inserted == 0
