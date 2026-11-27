"""Corpus versioning use cases."""

from __future__ import annotations

from datetime import datetime

from lankatax_ai.backend.corpus.store import CorpusStore
from lankatax_ai.backend.services.protocols import SessionFactory


async def finalize_ingest(
    *,
    session_factory: SessionFactory,
    run_id: str | None = None,
    ingested_at: datetime | None = None,
) -> int:
    """Finish an ingest run by advancing the corpus epoch.

    This is deliberately unconditional. If a run commits a document and then
    dies before finalization, the retry may process zero documents; it still
    needs to advance the epoch so old cache entries stop being served.
    """
    async with session_factory() as session:
        epoch = await CorpusStore(session).advance_epoch(
            run_id=run_id,
            ingested_at=ingested_at,
        )
        await session.commit()
        return epoch
