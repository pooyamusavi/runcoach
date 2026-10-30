"""FastAPI service exposing note extraction, Garmin import, analytics and the coach agent."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import ValidationError
from sqlalchemy.orm import Session, sessionmaker

from . import analytics
from .agent import CoachAgent
from .config import Settings, get_settings
from .db import TrainingNote, make_engine, make_session_factory
from .extract import Extractor, LLMExtractor, RuleExtractor
from .garmin import import_csv
from .llm import AnthropicLLM, LLMClient
from .schemas import AskIn, AskOut, NoteIn, Workout


def create_app(
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    if llm is None and settings.llm_enabled:
        llm = AnthropicLLM(settings.anthropic_api_key or "", settings.model)
    factory = session_factory or make_session_factory(make_engine(settings.database_url))
    extractor: Extractor = LLMExtractor(llm) if llm else RuleExtractor()
    extractor_name = "llm" if llm else "rules"

    app = FastAPI(title="RunCoach", version="0.1.0",
                  description="AI running coach: structured training logs, analytics and an agent.")

    def get_session() -> Iterator[Session]:
        session = factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "extractor": extractor_name}

    @app.post("/notes", response_model=Workout)
    def add_note(note: NoteIn, session: Session = Depends(get_session)) -> Workout:
        try:
            workout = extractor.extract(note.text, note.workout_date)
        except (ValidationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Extraction failed validation: {exc}") from exc
        session.add(
            TrainingNote(
                workout_date=workout.workout_date,
                raw_text=note.text,
                extracted=workout.model_dump(mode="json"),
                extractor=extractor_name,
            )
        )
        return workout

    @app.post("/import/garmin")
    async def import_garmin(file: UploadFile = File(...), session: Session = Depends(get_session)) -> dict[str, int]:
        text = (await file.read()).decode("utf-8-sig")
        try:
            result = import_csv(session, text)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"inserted": result.inserted, "updated": result.updated, "skipped": result.skipped}

    @app.get("/analytics/weekly")
    def weekly(weeks: int = 8, session: Session = Depends(get_session)) -> dict[str, Any]:
        return analytics.weekly_mileage(session, max(1, min(weeks, 52)))

    @app.get("/analytics/long-runs")
    def long_runs(weeks: int = 8, session: Session = Depends(get_session)) -> dict[str, Any]:
        return analytics.long_run_trend(session, max(1, min(weeks, 52)))

    @app.get("/analytics/training-load")
    def training_load(session: Session = Depends(get_session)) -> dict[str, Any]:
        return analytics.training_load(session)

    @app.post("/ask", response_model=AskOut)
    def ask(body: AskIn, session: Session = Depends(get_session)) -> AskOut:
        if llm is None:
            raise HTTPException(status_code=503, detail="Set ANTHROPIC_API_KEY to enable the coach agent.")
        result = CoachAgent(llm, settings.max_agent_steps).ask(session, body.question)
        return AskOut(answer=result.answer, tools_used=result.tools_used)

    return app
