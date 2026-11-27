from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import func, select, update

from lankatax_ai.backend.cache import PgResultCache
from lankatax_ai.backend.db.models import CacheEntry, CorpusState
from lankatax_ai.backend.rag.types import RetrievalResult

pytestmark = pytest.mark.db

EMBEDDING = [0.1] * 1536


def _result() -> RetrievalResult:
    return RetrievalResult(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content="PAYE is withheld by the employer.",
        score=0.91,
        vector_score=0.88,
        keyword_score=0.72,
        metadata={"from_url": "https://www.ird.gov.lk/paye"},
    )


async def _set_epoch(session, epoch: int) -> None:
    await session.execute(update(CorpusState).where(CorpusState.id == 1).values(epoch=epoch))
    await session.commit()


async def test_put_tags_entry_with_current_epoch(clean_session):
    await _set_epoch(clean_session, 4)
    cache = PgResultCache(clean_session)

    await cache.put("paye", EMBEDDING, [_result()])
    await clean_session.commit()

    entry = (await clean_session.execute(select(CacheEntry))).scalar_one()
    assert entry.corpus_epoch == 4


async def test_get_ignores_entries_from_previous_epochs(clean_session):
    await _set_epoch(clean_session, 1)
    cache = PgResultCache(clean_session)
    await cache.put("paye", EMBEDDING, [_result()])
    await clean_session.commit()

    await _set_epoch(clean_session, 2)

    assert await cache.get("paye", EMBEDDING) is None

    await cache.put("paye", EMBEDDING, [_result()])
    await clean_session.commit()

    cached = await cache.get("paye", EMBEDDING)
    assert cached is not None
    assert cached[0].content == "PAYE is withheld by the employer."


async def test_stats_and_expiry_only_consider_the_current_epoch(clean_session):
    old_entry = CacheEntry(query_text="old-1", response="[]", corpus_epoch=1)
    old_entry_2 = CacheEntry(query_text="old-2", response="[]", corpus_epoch=1)
    current_entry = CacheEntry(query_text="current", response="[]", corpus_epoch=2)
    clean_session.add_all([old_entry, old_entry_2, current_entry])
    await _set_epoch(clean_session, 2)

    cache = PgResultCache(clean_session)
    stats = await cache.get_stats()

    assert stats.total_entries == 1

    removed = await cache.invalidate_expired()
    await clean_session.commit()

    assert removed == 2
    remaining = await clean_session.scalar(select(func.count()).select_from(CacheEntry))
    assert remaining == 1
