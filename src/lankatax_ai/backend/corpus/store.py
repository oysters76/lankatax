"""Persistence for the singleton corpus version row."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, update
from sqlalchemy.ext.asyncio import AsyncSession

from lankatax_ai.backend.db.models import CorpusState


class CorpusStore:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def advance_epoch(
        self,
        *,
        run_id: str | None = None,
        ingested_at: datetime | None = None,
    ) -> int:
        """Advance the corpus epoch and return the new value."""
        stmt = (
            update(CorpusState)
            .where(CorpusState.id == 1)
            .values(
                epoch=CorpusState.epoch + 1,
                last_ingest_at=ingested_at if ingested_at is not None else func.now(),
                last_refresh_run_id=func.coalesce(
                    run_id,
                    CorpusState.last_refresh_run_id,
                ),
                updated_at=func.now(),
            )
            .returning(CorpusState.epoch)
        )
        epoch = await self._session.scalar(stmt)
        if epoch is None:
            raise RuntimeError("corpus_state row is missing")
        return epoch
