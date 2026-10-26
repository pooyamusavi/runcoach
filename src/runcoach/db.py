"""Database models and session handling (PostgreSQL in Docker, SQLite by default)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime

from sqlalchemy import JSON, Date, DateTime, Float, Integer, String, Text, UniqueConstraint, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


class Activity(Base):
    """One activity imported from a Garmin Connect CSV export."""

    __tablename__ = "activities"
    __table_args__ = (UniqueConstraint("start_time", "activity_type", name="uq_activity_natural_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    start_time: Mapped[datetime] = mapped_column(DateTime, index=True)
    activity_type: Mapped[str] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(String(255), default="")
    distance_km: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_hr: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_hr: Mapped[int | None] = mapped_column(Integer, nullable=True)
    avg_pace_min_per_km: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_ascent_m: Mapped[float | None] = mapped_column(Float, nullable=True)


class TrainingNote(Base):
    """A free-text note plus its validated structured extraction."""

    __tablename__ = "training_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    workout_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    raw_text: Mapped[str] = mapped_column(Text)
    extracted: Mapped[dict[str, object]] = mapped_column(JSON)
    extractor: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC).replace(tzinfo=None))


def make_engine(database_url: str) -> Engine:
    kwargs: dict[str, object] = {}
    if database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in database_url:
            kwargs["poolclass"] = StaticPool  # one shared in-memory DB across threads (tests)
    engine = create_engine(database_url, **kwargs)
    Base.metadata.create_all(engine)
    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
