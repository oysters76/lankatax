from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

import pytest

from lankatax_ai.backend.cache import PgResultCache
from lankatax_ai.backend.db.models import CorpusState
from lankatax_ai.backend.rag.types import RetrievalResult
from lankatax_ai.backend.services.corpus import (
    advance_corpus_epoch,
    advance_corpus_epoch_after_ingest,
)

pytestmark = pytest.mark.db

EMBEDDING = [0.1] * 1536


def _result() -> RetrievalResult:
    return RetrievalResult(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content="PAYE is withheld by the employer.",
        score=0.9,
        vector_score=0.9,
        keyword_score=0.8,
        metadata={},
    )


def _factory(session):
    @asynccontextmanager
    async def factory():
        yield session

    return factory


async def test_advance_corpus_epoch_updates_state(clean_session):
    epoch = await advance_corpus_epoch(
        session_factory=_factory(clean_session),
        last_refresh_run_id="run-1",
    )

    assert epoch == 1
    state = await clean_session.get(CorpusState, 1)
    assert state is not None
    await clean_session.refresh(state)
    assert state.epoch == 1
    assert state.last_refresh_run_id == "run-1"
    assert state.last_ingest_at is not None


async def test_advance_after_ingest_is_a_noop_without_documents(clean_session):
    epoch = await advance_corpus_epoch_after_ingest(
        session_factory=_factory(clean_session),
        docs_processed=0,
    )

    assert epoch is None
    state = await clean_session.get(CorpusState, 1)
    assert state is not None
    assert state.epoch == 0


async def test_advance_after_ingest_bumps_with_documents(clean_session):
    epoch = await advance_corpus_epoch_after_ingest(
        session_factory=_factory(clean_session),
        docs_processed=2,
        last_refresh_run_id="run-2",
    )

    assert epoch == 1


async def test_epoch_bump_hides_previous_cache_entries(clean_session):
    cache = PgResultCache(clean_session)
    await cache.put("paye", EMBEDDING, [_result()])
    await clean_session.commit()

    assert await cache.get("paye", EMBEDDING) is not None

    await advance_corpus_epoch(session_factory=_factory(clean_session))

    assert await cache.get("paye", EMBEDDING) is None
