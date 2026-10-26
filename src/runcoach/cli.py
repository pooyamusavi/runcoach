"""Command-line entry point: import Garmin data, extract a note, ask the coach, run evals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import CoachAgent
from .config import get_settings
from .db import make_engine, make_session_factory, session_scope
from .extract import LLMExtractor, RuleExtractor
from .garmin import import_csv
from .llm import AnthropicLLM


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="runcoach")
    sub = parser.add_subparsers(dest="command", required=True)
    p_import = sub.add_parser("import", help="Import a Garmin Activities.csv export")
    p_import.add_argument("csv_path", type=Path)
    p_extract = sub.add_parser("extract", help="Extract structured data from a note")
    p_extract.add_argument("text")
    p_extract.add_argument("--rules", action="store_true", help="Use the offline rule baseline")
    p_ask = sub.add_parser("ask", help="Ask the coaching agent a question")
    p_ask.add_argument("question")
    args = parser.parse_args(argv)

    settings = get_settings()
    factory = make_session_factory(make_engine(settings.database_url))
    llm = AnthropicLLM(settings.anthropic_api_key, settings.model) if settings.anthropic_api_key else None

    if args.command == "import":
        with session_scope(factory) as session:
            imported = import_csv(session, args.csv_path.read_text(encoding="utf-8-sig"))
        print(json.dumps(imported.__dict__))
        return 0

    if args.command == "extract":
        extractor = RuleExtractor() if args.rules or llm is None else LLMExtractor(llm)
        print(extractor.extract(args.text).model_dump_json(indent=2))
        return 0

    if args.command == "ask":
        if llm is None:
            print("Set ANTHROPIC_API_KEY to use the coach agent.", file=sys.stderr)
            return 2
        with session_scope(factory) as session:
            answer = CoachAgent(llm, settings.max_agent_steps).ask(session, args.question)
        print(answer.answer)
        print(f"\n[tools used: {', '.join(answer.tools_used) or 'none'}]")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
