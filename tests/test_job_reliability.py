from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from sqlalchemy import func, select

from lankatax_ai.backend.db.models import Chunk as ChunkModel
from lankatax_ai.backend.db.models import Document
from lankatax_ai.backend.documents import DocumentStore
from lankatax_ai.backend.ingestion.pipeline import ingest_batches, ingest_document
from lankatax_ai.backend.ingestion.store import IngestionStore
from lankatax_ai.backend.ingestion.types import ChunkConfig, EmbeddingPolicy
from tests.helpers import make_document

pytestmark = pytest.mark.db

POLICY = EmbeddingPolicy()


def _model() -> AsyncMock:
    model = AsyncMock()
    model.aembed_documents = AsyncMock(side_effect=lambda texts: [[0.1] * 1536 for _ in texts])
    return model


async def _insert(session, **overrides):
    doc = make_document(**overrides)
    store = DocumentStore(session)
    await store.save([doc])
    await store.commit()
    document_id = (
        await session.execute(select(Document.id).where(Document.url == doc.url))
    ).scalar_one()
    return await store.get(document_id), document_id


async def _chunk_count(session, document_id) -> int:
    return (
        await session.scalar(
            select(func.count())
            .select_from(ChunkModel)
            .where(ChunkModel.document_id == document_id)
        )
    ) or 0


async def test_reingest_replaces_chunks_without_duplication(clean_session):
    doc, doc_id = await _insert(
        clean_session,
        url="https://www.ird.gov.lk/rebuild",
        raw_text="PAYE is deducted by the employer. Employees file annual returns.",
        raw_html="<p>PAYE is deducted by the employer.</p>",
    )
    model = _model()

    first_count, _ = await ingest_document(
        clean_session,
        doc,
        ChunkConfig(),
        model,
        POLICY,
    )
    await clean_session.commit()

    refreshed = await DocumentStore(clean_session).get(doc_id)
    assert refreshed is not None
    second_count, _ = await ingest_document(
        clean_session,
        refreshed,
        ChunkConfig(),
        model,
        POLICY,
    )
    await clean_session.commit()

    assert first_count == second_count
    assert await _chunk_count(clean_session, doc_id) == second_count
    assert (await clean_session.scalar(select(func.count()).select_from(Document))) == 1


async def test_budget_exhaustion_writes_no_chunks(clean_session):
    _doc, doc_id = await _insert(
        clean_session,
        url="https://www.ird.gov.lk/budget",
        raw_text="A long enough tax document to produce more than one token.",
        raw_html="<p>A long enough tax document.</p>",
    )
    model = _model()

    results = []
    async for batch in ingest_batches(
        IngestionStore(clean_session),
        [doc_id],
        ChunkConfig(),
        model,
        POLICY,
        max_tokens=1,
    ):
        results.append(batch)

    assert results
    assert results[-1].budget_exhausted is True
    assert await _chunk_count(clean_session, doc_id) == 0
    assert (await clean_session.scalar(select(func.count()).select_from(ChunkModel))) == 0


async def test_scrape_upsert_is_idempotent(clean_session):
    store = DocumentStore(clean_session)
    url = "https://www.ird.gov.lk/upsert"

    await store.save([make_document(url=url, content_hash="old")])
    await store.commit()
    await store.save([make_document(url=url, content_hash="new")])
    await store.commit()

    count = await clean_session.scalar(select(func.count()).select_from(Document))
    assert count == 1
    hashes = await store.hash_map([url])
    assert hashes == {url: "new"}


async def test_ingest_iterator_can_stop_after_a_committed_document(clean_session):
    _first, first_id = await _insert(
        clean_session,
        url="https://www.ird.gov.lk/one",
        raw_text="PAYE is deducted by the employer and employees file returns.",
        raw_html="<p>PAYE is deducted.</p>",
    )
    _second, second_id = await _insert(
        clean_session,
        url="https://www.ird.gov.lk/two",
        raw_text="VAT applies to taxable supplies made in Sri Lanka.",
        raw_html="<p>VAT applies.</p>",
    )
    model = _model()

    batches = ingest_batches(
        IngestionStore(clean_session),
        [first_id, second_id],
        ChunkConfig(),
        model,
        POLICY,
    )
    first_batch = await anext(batches)
    await batches.aclose()

    assert first_batch.error is None
    assert await _chunk_count(clean_session, first_id) > 0
    assert await _chunk_count(clean_session, second_id) == 0
