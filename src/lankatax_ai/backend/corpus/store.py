"""Persistence for the singleton corpus version row."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func, update
from sqlalchemy.ext.asyncio import AsyncSession

from lankatax_ai.backend.db.models import CorpusState


class CorpusStore:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def advance_epoch(
        self,
        *,
        last_refresh_run_id: str | None = None,
        ingested_at: datetime | None = None,
    ) -> int:
        """Advance the corpus epoch and return the new value."""
        values: dict[str, Any] = {
            "epoch": CorpusState.epoch + 1,
            "last_ingest_at": ingested_at if ingested_at is not None else func.now(),
            "updated_at": func.now(),
        }
        if last_refresh_run_id is not None:
            values["last_refresh_run_id"] = last_refresh_run_id

        stmt = (
            update(CorpusState)
            .where(CorpusState.id == 1)
            .values(**values)
            .returning(CorpusState.epoch)
        )
        epoch = await self._session.scalar(stmt)
        if epoch is None:
            raise RuntimeError("corpus_state row is missing")
        return epoch
