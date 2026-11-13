from unittest.mock import AsyncMock

from sqlalchemy import text

from lankatax_ai.backend.ingestion.pipeline import (
    get_ingestion_status,
    ingest_document,
    ingest_pending,
)
from lankatax_ai.backend.ingestion.types import ChunkConfig, ChunkStrategy
from lankatax_ai.backend.scraper.storage import get_document_by_id, save_documents
from tests.helpers import make_document_record


def _make_mock_model():
    mock = AsyncMock()
    mock.aembed_documents = AsyncMock(side_effect=lambda texts: [[0.1] * 1536 for _ in texts])
    return mock


async def _insert_doc(session, **overrides):
    doc = make_document_record(**overrides)
    await save_documents(session, [doc])
    row = await session.execute(text("SELECT id FROM documents WHERE url = :url"), {"url": doc.url})
    doc_id = row.mappings().fetchone()["id"]
    return await get_document_by_id(session, doc_id)


async def test_ingest_document(clean_session):
    doc = await _insert_doc(
        clean_session,
        raw_text="PAYE tax is deducted by employer. Employees must file returns.",
        raw_html="<p>PAYE tax is deducted by employer. Employees must file returns.</p>",
    )
    model = _make_mock_model()
    config = ChunkConfig(strategy=ChunkStrategy.structural)

    chunk_count = await ingest_document(clean_session, doc, config, model)

    assert chunk_count > 0
    result = await clean_session.execute(
        text("SELECT COUNT(*) AS cnt FROM chunks WHERE document_id = :id"),
        {"id": doc.id},
    )
    assert result.mappings().fetchone()["cnt"] == chunk_count

    refreshed = await get_document_by_id(clean_session, doc.id)
    assert refreshed.ingested_hash == doc.content_hash


async def test_ingest_document_replaces_old_chunks(clean_session):
    doc = await _insert_doc(
        clean_session,
        raw_text="Original content about VAT rates.",
        raw_html="<p>Original content about VAT rates.</p>",
    )
    model = _make_mock_model()
    config = ChunkConfig(strategy=ChunkStrategy.structural)

    await ingest_document(clean_session, doc, config, model)

    doc.raw_text = "Updated content about income tax rates and PAYE deductions."
    doc.raw_html = "<p>Updated content about income tax rates and PAYE deductions.</p>"
    doc.content_hash = "updated_hash"
    second_count = await ingest_document(clean_session, doc, config, model)

    result = await clean_session.execute(
        text("SELECT COUNT(*) AS cnt FROM chunks WHERE document_id = :id"),
        {"id": doc.id},
    )
    assert result.mappings().fetchone()["cnt"] == second_count


async def test_ingest_pending_skips_ingested(clean_session):
    await _insert_doc(
        clean_session,
        url="https://www.ird.gov.lk/pending1",
        raw_text="Pending doc content.",
        raw_html="<p>Pending doc content.</p>",
    )
    doc2 = await _insert_doc(
        clean_session,
        url="https://www.ird.gov.lk/ingested1",
        raw_text="Already ingested content.",
        raw_html="<p>Already ingested content.</p>",
    )
    await clean_session.execute(
        text("UPDATE documents SET ingested_hash = content_hash WHERE id = :id"),
        {"id": doc2.id},
    )
    await clean_session.commit()

    model = _make_mock_model()
    config = ChunkConfig(strategy=ChunkStrategy.structural)
    result = await ingest_pending(clean_session, config, model)

    assert result.docs_processed == 1
    assert result.chunks_created > 0
    assert len(result.errors) == 0


async def test_ingest_pending_empty(clean_session):
    model = _make_mock_model()
    config = ChunkConfig(strategy=ChunkStrategy.structural)
    result = await ingest_pending(clean_session, config, model)

    assert result.docs_processed == 0
    assert result.chunks_created == 0
    assert len(result.errors) == 0


async def test_get_ingestion_status(clean_session):
    await _insert_doc(
        clean_session,
        url="https://www.ird.gov.lk/status1",
        raw_text="Pending.",
        raw_html="<p>Pending.</p>",
    )
    doc2 = await _insert_doc(
        clean_session,
        url="https://www.ird.gov.lk/status2",
        raw_text="Ingested.",
        raw_html="<p>Ingested.</p>",
    )
    await clean_session.execute(
        text("UPDATE documents SET ingested_hash = content_hash WHERE id = :id"),
        {"id": doc2.id},
    )
    await clean_session.commit()

    status = await get_ingestion_status(clean_session)
    assert status.total_documents == 2
    assert status.ingested == 1
    assert status.pending == 1
