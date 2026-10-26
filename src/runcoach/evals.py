"""Field-level evaluation of an extractor against a labelled set of training notes.

Each expected field is scored independently:
* numbers match within a relative tolerance (default 3%), so 5:40/km = 5.67 passes,
* enums and integers must match exactly,
* symptom lists are scored by recall of the expected symptoms (fuzzy word overlap).
Fields absent from ``expected`` are not scored, so the set rewards correct extraction
without penalising extra, correct detail.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .extract import Extractor

NUMERIC_FIELDS = {"distance_km", "duration_min", "avg_pace_min_per_km"}


def _close(a: float | None, b: float, tol: float) -> bool:
    return a is not None and abs(a - b) <= max(tol * abs(b), 0.05)


def _symptom_recall(actual: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0
    actual_words = [set(s.lower().replace("-", " ").split()) for s in actual]
    hits = 0
    for exp in expected:
        exp_words = set(exp.lower().split())
        if any(len(exp_words & a) >= max(1, len(exp_words) - 1) for a in actual_words):
            hits += 1
    return hits / len(expected)


@dataclass
class EvalReport:
    per_field: dict[str, list[float]] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    def field_accuracy(self) -> dict[str, float]:
        return {k: round(sum(v) / len(v), 3) for k, v in sorted(self.per_field.items())}

    def overall(self) -> float:
        scores = [s for v in self.per_field.values() for s in v]
        return round(sum(scores) / len(scores), 3) if scores else 0.0


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate(extractor: Extractor, cases: list[dict[str, Any]], tol: float = 0.03) -> EvalReport:
    report = EvalReport()
    for case in cases:
        try:
            workout = extractor.extract(case["text"]).model_dump(mode="json")
        except Exception as exc:
            report.errors.append({"id": case["id"], "error": str(exc)})
            for key in case["expected"]:
                report.per_field.setdefault(key, []).append(0.0)
            continue
        for key, expected in case["expected"].items():
            actual = workout.get(key)
            if key == "symptoms":
                score = _symptom_recall(actual or [], expected)
            elif key in NUMERIC_FIELDS:
                score = 1.0 if _close(actual, float(expected), tol) else 0.0
            else:
                score = 1.0 if actual == expected else 0.0
            report.per_field.setdefault(key, []).append(score)
            if score < 1.0:
                report.failures.append({"id": case["id"], "field": key, "expected": expected, "actual": actual})
    return report
