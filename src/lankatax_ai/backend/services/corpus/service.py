"""Corpus versioning use cases."""

from __future__ import annotations

from datetime import datetime

from lankatax_ai.backend.corpus.store import CorpusStore
from lankatax_ai.backend.services.protocols import SessionFactory


async def advance_corpus_epoch(
    *,
    session_factory: SessionFactory,
    last_refresh_run_id: str | None = None,
    ingested_at: datetime | None = None,
) -> int:
    """Advance the corpus epoch and commit it."""
    async with session_factory() as session:
        epoch = await CorpusStore(session).advance_epoch(
            last_refresh_run_id=last_refresh_run_id,
            ingested_at=ingested_at,
        )
        await session.commit()
        return epoch


async def advance_corpus_epoch_after_ingest(
    *,
    session_factory: SessionFactory,
    docs_processed: int,
    last_refresh_run_id: str | None = None,
    ingested_at: datetime | None = None,
) -> int | None:
    """Advance only when at least one document was committed."""
    if docs_processed <= 0:
        return None
    return await advance_corpus_epoch(
        session_factory=session_factory,
        last_refresh_run_id=last_refresh_run_id,
        ingested_at=ingested_at,
    )
