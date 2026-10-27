"""Runtime configuration, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root: src/runcoach/config.py -> src/runcoach -> src -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_DATABASE_URL = f"sqlite:///{PROJECT_ROOT / 'runcoach.db'}"


@dataclass(frozen=True)
class Settings:
    database_url: str
    anthropic_api_key: str | None
    model: str
    max_agent_steps: int

    @property
    def llm_enabled(self) -> bool:
        return bool(self.anthropic_api_key)


def get_settings() -> Settings:
    return Settings(
        database_url=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        model=os.getenv("RUNCOACH_MODEL", "claude-sonnet-5-5"),
        max_agent_steps=int(os.getenv("RUNCOACH_MAX_AGENT_STEPS", "6")),
    )
