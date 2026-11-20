import dataclasses
import logging
from collections.abc import Callable
from uuid import UUID

from langchain_openai import OpenAIEmbeddings
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from lankatax_ai.backend.db.models import Document
from lankatax_ai.backend.ingestion.chunker import chunk_document
from lankatax_ai.backend.ingestion.embedder import embed_chunks
from lankatax_ai.backend.ingestion.types import ChunkConfig
from lankatax_ai.backend.ingestion.vector_store import (
    delete_chunks_by_document_id,
    store_chunks,
)
from lankatax_ai.backend.scraper.storage import (
    get_all_document_ids,
    get_document_by_id,
    get_pending_document_ids,
    update_ingested_hash,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class IngestionResult:
    docs_processed: int = 0
    chunks_created: int = 0
    errors: list[str] = dataclasses.field(default_factory=list)
    tokens_used: int = 0


@dataclasses.dataclass
class IngestionStatusRecord:
    total_documents: int
    pending: int
    ingested: int


async def ingest_document(
    session: AsyncSession,
    doc: Document,
    config: ChunkConfig,
    model: OpenAIEmbeddings,
) -> tuple[int, int]:
    logger.info("Ingesting doc %s — %s", doc.id, doc.title or doc.url)
    chunks = await chunk_document(doc, config, model)
    embedded = await embed_chunks(model, chunks)
    tokens_used = sum([c.token_count for c in chunks])
    deleted = await delete_chunks_by_document_id(session, doc.id)
    logger.debug("Deleted %d old chunks for doc %s", deleted, doc.id)
    await store_chunks(session, doc.id, embedded)
    await update_ingested_hash(session, doc.id, doc.content_hash)
    await session.flush()
    logger.info("Done doc %s — %d chunks, %d tokens", doc.id, len(embedded), tokens_used)
    return len(embedded), tokens_used


async def _ingest_ids(
    session: AsyncSession,
    doc_ids: list[UUID],
    config: ChunkConfig,
    model: OpenAIEmbeddings,
    on_progress: Callable[[UUID, int, str, int, str | None], None] | None = None,
) -> IngestionResult:
    logger.info("Starting ingestion of %d documents", len(doc_ids))
    result = IngestionResult()
    for doc_id in doc_ids:
        try:
            doc = await get_document_by_id(session, doc_id)
            if doc is None:
                logger.warning("Document %s not found, skipping", doc_id)
                result.errors.append(f"Document {doc_id} not found")
                if on_progress:
                    on_progress(doc_id, 0, str(doc_id), 0, "not found")
                continue
            chunk_count, tokens_used = await ingest_document(session, doc, config, model)
            doc_title = doc.title or doc.url
            await session.commit()
            result.docs_processed += 1
            result.chunks_created += chunk_count
            result.tokens_used += tokens_used
            if on_progress:
                on_progress(doc_id, chunk_count, doc_title, tokens_used, None)
        except Exception as e:
            logger.exception("Failed to ingest doc %s", doc_id)
            result.errors.append(f"{doc_id}: {e}")
            await session.rollback()
            if on_progress:
                on_progress(doc_id, 0, str(doc_id), 0, str(e))
    logger.info(
        "Ingestion complete — %d docs, %d chunks, %d tokens",
        result.docs_processed,
        result.chunks_created,
        result.tokens_used,
    )
    return result


async def ingest_pending(
    session: AsyncSession,
    config: ChunkConfig,
    model: OpenAIEmbeddings,
    on_progress: Callable[[UUID, int], None] | None = None,
) -> IngestionResult:
    doc_ids = await get_pending_document_ids(session)
    logger.info("Found %d pending documents", len(doc_ids))
    return await _ingest_ids(session, doc_ids, config, model, on_progress)


async def ingest_all(
    session: AsyncSession,
    config: ChunkConfig,
    model: OpenAIEmbeddings,
    on_progress: Callable[[UUID, int], None] | None = None,
) -> IngestionResult:
    doc_ids = await get_all_document_ids(session)
    logger.info("Found %d total documents", len(doc_ids))
    return await _ingest_ids(session, doc_ids, config, model, on_progress)


async def get_ingestion_status(session: AsyncSession) -> IngestionStatusRecord:
    is_ingested = Document.ingested_hash.is_not(None) & (
        Document.content_hash == Document.ingested_hash
    )
    stmt = select(
        func.count().label("total_documents"),
        func.count(case((is_ingested, 1))).label("ingested"),
        func.count(case((~is_ingested, 1))).label("pending"),
    )
    row = (await session.execute(stmt)).one()
    return IngestionStatusRecord(
        total_documents=row.total_documents,
        ingested=row.ingested,
        pending=row.pending,
    )
