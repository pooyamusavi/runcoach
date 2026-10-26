"""Run the extraction evals.

    python evals/run_evals.py            # rule baseline (offline)
    python evals/run_evals.py --llm      # Claude extractor (needs ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from runcoach.config import get_settings
from runcoach.evals import evaluate, load_cases
from runcoach.extract import Extractor, LLMExtractor, RuleExtractor
from runcoach.llm import AnthropicLLM

CASES = Path(__file__).with_name("notes.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--show-failures", action="store_true")
    args = parser.parse_args()

    extractor: Extractor
    if args.llm:
        settings = get_settings()
        if not settings.anthropic_api_key:
            print("ANTHROPIC_API_KEY is not set", file=sys.stderr)
            return 2
        extractor = LLMExtractor(AnthropicLLM(settings.anthropic_api_key, settings.model))
        name = f"llm ({settings.model})"
    else:
        extractor, name = RuleExtractor(), "rule baseline"

    report = evaluate(extractor, load_cases(CASES))
    print(f"Extractor: {name}")
    print(f"Overall field accuracy: {report.overall():.1%}")
    for key, acc in report.field_accuracy().items():
        print(f"  {key:<22} {acc:.1%}")
    if report.errors:
        print(f"Errors: {len(report.errors)}")
    if args.show_failures:
        print(json.dumps(report.failures, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
