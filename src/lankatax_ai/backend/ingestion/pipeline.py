"""Document ingestion: chunk, embed and store.

The public functions here keep the signature the frozen CLI uses. New callers go
through :mod:`lankatax_ai.backend.services.ingest`, which owns sessions and takes
its collaborators from deps.
"""

import logging
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from lankatax_ai.backend.db.models import Document
from lankatax_ai.backend.ingestion.chunker import chunk_document
from lankatax_ai.backend.ingestion.embedder import embed_chunks
from lankatax_ai.backend.ingestion.store import IngestionStore
from lankatax_ai.backend.ingestion.types import (
    ChunkConfig,
    DocumentEmbeddings,
    EmbeddingPolicy,
    IngestionRepository,
    IngestionResult,
    IngestionStatus,
    ProgressCallback,
)

logger = logging.getLogger(__name__)

# Kept for callers that imported the old name.
IngestionStatusRecord = IngestionStatus


async def ingest_one(
    repo: IngestionRepository,
    doc: Document,
    config: ChunkConfig,
    embeddings: DocumentEmbeddings,
    policy: EmbeddingPolicy,
) -> tuple[int, int]:
    """Chunk, embed and persist one document. Returns (chunk count, tokens)."""
    logger.info("Ingesting doc %s — %s", doc.id, doc.title or doc.url)
    chunks = await chunk_document(doc, config, embeddings)
    embedded = await embed_chunks(embeddings, chunks, policy=policy)
    tokens_used = sum(c.token_count for c in chunks)
    deleted = await repo.replace_chunks(doc.id, embedded)
    logger.debug("Deleted %d old chunks for doc %s", deleted, doc.id)
    await repo.mark_ingested(doc.id, doc.content_hash)
    logger.info("Done doc %s — %d chunks, %d tokens", doc.id, len(embedded), tokens_used)
    return len(embedded), tokens_used


async def ingest_documents(
    repo: IngestionRepository,
    doc_ids: list[UUID],
    config: ChunkConfig,
    embeddings: DocumentEmbeddings,
    policy: EmbeddingPolicy,
    on_progress: ProgressCallback | None = None,
) -> IngestionResult:
    """Ingest each document, committing per document and isolating failures."""
    logger.info("Starting ingestion of %d documents", len(doc_ids))
    result = IngestionResult()
    for doc_id in doc_ids:
        try:
            doc = await repo.get_document(doc_id)
            if doc is None:
                logger.warning("Document %s not found, skipping", doc_id)
                result.errors.append(f"Document {doc_id} not found")
                if on_progress:
                    on_progress(doc_id, 0, str(doc_id), 0, "not found")
                continue
            chunk_count, tokens_used = await ingest_one(repo, doc, config, embeddings, policy)
            doc_title = doc.title or doc.url
            await repo.commit()
            result.docs_processed += 1
            result.chunks_created += chunk_count
            result.tokens_used += tokens_used
            if on_progress:
                on_progress(doc_id, chunk_count, doc_title, tokens_used, None)
        except Exception as e:
            logger.exception("Failed to ingest doc %s", doc_id)
            result.errors.append(f"{doc_id}: {e}")
            await repo.rollback()
            if on_progress:
                on_progress(doc_id, 0, str(doc_id), 0, str(e))
    logger.info(
        "Ingestion complete — %d docs, %d chunks, %d tokens",
        result.docs_processed,
        result.chunks_created,
        result.tokens_used,
    )
    return result


async def ingest_document(
    session: AsyncSession,
    doc: Document,
    config: ChunkConfig,
    model: DocumentEmbeddings,
    policy: EmbeddingPolicy,
) -> tuple[int, int]:
    """Ingest one already-loaded document. The caller owns the commit."""
    return await ingest_one(IngestionStore(session), doc, config, model, policy)


async def ingest_pending(
    session: AsyncSession,
    config: ChunkConfig,
    model: DocumentEmbeddings,
    policy: EmbeddingPolicy,
    on_progress: ProgressCallback | None = None,
) -> IngestionResult:
    repo = IngestionStore(session)
    doc_ids = await repo.pending_document_ids()
    logger.info("Found %d pending documents", len(doc_ids))
    return await ingest_documents(repo, doc_ids, config, model, policy, on_progress)


async def ingest_all(
    session: AsyncSession,
    config: ChunkConfig,
    model: DocumentEmbeddings,
    policy: EmbeddingPolicy,
    on_progress: ProgressCallback | None = None,
) -> IngestionResult:
    repo = IngestionStore(session)
    doc_ids = await repo.all_document_ids()
    logger.info("Found %d total documents", len(doc_ids))
    return await ingest_documents(repo, doc_ids, config, model, policy, on_progress)


async def get_ingestion_status(session: AsyncSession) -> IngestionStatus:
    return await IngestionStore(session).status()
