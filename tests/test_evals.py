from __future__ import annotations

from pathlib import Path

from runcoach.evals import _symptom_recall, evaluate, load_cases
from runcoach.extract import RuleExtractor

CASES = Path(__file__).resolve().parents[1] / "evals" / "notes.jsonl"


def test_symptom_recall_is_fuzzy() -> None:
    assert _symptom_recall(["tight calf", "fatigue"], ["calf tightness"]) == 1.0
    assert _symptom_recall([], ["knee pain"]) == 0.0


def test_rule_baseline_runs_on_eval_set() -> None:
    report = evaluate(RuleExtractor(), load_cases(CASES))
    assert not report.errors
    assert 0.3 < report.overall() < 1.0  # a baseline: useful, but leaves room for the LLM
