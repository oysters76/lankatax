import dataclasses
from collections.abc import Callable
from uuid import UUID

from langchain_openai import OpenAIEmbeddings
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from lankatax_ai.backend.ingestion.chunker import chunk_document
from lankatax_ai.backend.ingestion.embedder import embed_chunks
from lankatax_ai.backend.ingestion.types import ChunkConfig
from lankatax_ai.backend.ingestion.vector_store import (
    delete_chunks_by_document_id,
    store_chunks,
)
from lankatax_ai.backend.scraper.storage import (
    DocumentRecord,
    get_all_document_ids,
    get_document_by_id,
    get_pending_document_ids,
    update_ingested_hash,
)


@dataclasses.dataclass
class IngestionResult:
    docs_processed: int = 0
    chunks_created: int = 0
    errors: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class IngestionStatusRecord:
    total_documents: int
    pending: int
    ingested: int


async def ingest_document(
    session: AsyncSession,
    doc: DocumentRecord,
    config: ChunkConfig,
    model: OpenAIEmbeddings,
) -> int:
    chunks = await chunk_document(doc, config, model)
    embedded = await embed_chunks(model, chunks)
    await delete_chunks_by_document_id(session, doc.id)
    await store_chunks(session, doc.id, embedded)
    await update_ingested_hash(session, doc.id, doc.content_hash)
    return len(embedded)


async def _ingest_ids(
    session: AsyncSession,
    doc_ids: list[UUID],
    config: ChunkConfig,
    model: OpenAIEmbeddings,
    on_progress: Callable[[UUID, int], None] | None = None,
) -> IngestionResult:
    result = IngestionResult()
    for doc_id in doc_ids:
        try:
            doc = await get_document_by_id(session, doc_id)
            if doc is None:
                result.errors.append(f"Document {doc_id} not found")
                continue
            chunk_count = await ingest_document(session, doc, config, model)
            result.docs_processed += 1
            result.chunks_created += chunk_count
            if on_progress:
                on_progress(doc_id, chunk_count)
        except Exception as e:  # noqa: BLE001
            result.errors.append(f"{doc_id}: {e}")
    return result


async def ingest_pending(
    session: AsyncSession,
    config: ChunkConfig,
    model: OpenAIEmbeddings,
    on_progress: Callable[[UUID, int], None] | None = None,
) -> IngestionResult:
    doc_ids = await get_pending_document_ids(session)
    return await _ingest_ids(session, doc_ids, config, model, on_progress)


async def ingest_all(
    session: AsyncSession,
    config: ChunkConfig,
    model: OpenAIEmbeddings,
    on_progress: Callable[[UUID, int], None] | None = None,
) -> IngestionResult:
    doc_ids = await get_all_document_ids(session)
    return await _ingest_ids(session, doc_ids, config, model, on_progress)


async def get_ingestion_status(session: AsyncSession) -> IngestionStatusRecord:
    query = text("""
        SELECT
            COUNT(*) AS total_documents,
            COUNT(CASE WHEN ingested_hash IS NOT NULL AND content_hash = ingested_hash THEN 1 END) AS ingested,
            COUNT(CASE WHEN ingested_hash IS NULL OR content_hash != ingested_hash THEN 1 END) AS pending
        FROM documents
    """)
    result = await session.execute(query)
    row = result.mappings().fetchone()
    return IngestionStatusRecord(**row)
