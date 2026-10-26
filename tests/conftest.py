from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from runcoach.db import make_engine, make_session_factory

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = ROOT / "sample_data" / "garmin_activities_sample.csv"


@pytest.fixture
def session_factory() -> sessionmaker[Session]:
    return make_session_factory(make_engine("sqlite:///:memory:"))


@pytest.fixture
def session(session_factory: sessionmaker[Session]) -> Session:
    s = session_factory()
    yield s
    s.close()


@pytest.fixture
def sample_csv() -> str:
    return SAMPLE_CSV.read_text(encoding="utf-8")
