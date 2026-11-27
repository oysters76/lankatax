"""Document ingestion: chunk, embed and store.

The public functions here keep the signature the frozen CLI uses. New callers go
through :mod:`lankatax_ai.backend.services.ingest`, which owns sessions and takes
its collaborators from deps.
"""

import logging
from collections.abc import AsyncIterator
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
    IngestBatch,
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
    *,
    remaining_tokens: int | None = None,
) -> tuple[int, int, bool]:
    """Chunk, embed and persist one document.

    Returns ``(chunk count, tokens, budget_exhausted)``. The budget check happens
    after chunking and before embedding, so an over-budget document spends no
    embedding tokens and writes nothing.
    """
    logger.info("Ingesting doc %s — %s", doc.id, doc.title or doc.url)
    chunks = await chunk_document(doc, config, embeddings)
    tokens_used = sum(c.token_count for c in chunks)
    if remaining_tokens is not None and tokens_used > remaining_tokens:
        logger.warning(
            "Stopping ingest for doc %s: %d tokens exceeds remaining budget %d",
            doc.id,
            tokens_used,
            remaining_tokens,
        )
        return 0, 0, True

    embedded = await embed_chunks(embeddings, chunks, policy=policy)
    deleted = await repo.replace_chunks(doc.id, embedded)
    logger.debug("Deleted %d old chunks for doc %s", deleted, doc.id)
    await repo.mark_ingested(doc.id, doc.content_hash)
    logger.info("Done doc %s — %d chunks, %d tokens", doc.id, len(embedded), tokens_used)
    return len(embedded), tokens_used, False


async def ingest_batches(
    repo: IngestionRepository,
    doc_ids: list[UUID],
    config: ChunkConfig,
    embeddings: DocumentEmbeddings,
    policy: EmbeddingPolicy,
    *,
    max_tokens: int | None = None,
) -> AsyncIterator[IngestBatch]:
    """Yield one result per document, committing each successful document.

    Consumers stop by breaking/closing the iterator. The token budget is checked
    before embedding; an over-budget document yields a terminal sentinel.
    """
    logger.info("Starting ingestion of %d documents", len(doc_ids))
    total_tokens = 0

    for doc_id in doc_ids:
        try:
            doc = await repo.get_document(doc_id)
            if doc is None:
                logger.warning("Document %s not found, skipping", doc_id)
                yield IngestBatch(doc_id, str(doc_id), 0, 0, "not found")
                continue

            document_id = doc.id
            title = doc.title or doc.url
            remaining = None if max_tokens is None else max_tokens - total_tokens
            if remaining is not None and remaining <= 0:
                yield IngestBatch(
                    document_id,
                    title,
                    0,
                    0,
                    "token budget exhausted",
                    budget_exhausted=True,
                )
                return

            chunk_count, tokens_used, budget_exhausted = await ingest_one(
                repo,
                doc,
                config,
                embeddings,
                policy,
                remaining_tokens=remaining,
            )
            if budget_exhausted:
                yield IngestBatch(
                    document_id,
                    title,
                    0,
                    0,
                    "token budget exhausted",
                    budget_exhausted=True,
                )
                return

            await repo.commit()
            total_tokens += tokens_used
            yield IngestBatch(document_id, title, chunk_count, tokens_used)
        except Exception as error:
            logger.exception("Failed to ingest doc %s", doc_id)
            await repo.rollback()
            yield IngestBatch(doc_id, str(doc_id), 0, 0, str(error))


async def ingest_documents(
    repo: IngestionRepository,
    doc_ids: list[UUID],
    config: ChunkConfig,
    embeddings: DocumentEmbeddings,
    policy: EmbeddingPolicy,
    on_progress: ProgressCallback | None = None,
    *,
    max_tokens: int | None = None,
) -> IngestionResult:
    """Collect per-document batches into one result (legacy CLI path)."""
    result = IngestionResult()
    async for batch in ingest_batches(
        repo,
        doc_ids,
        config,
        embeddings,
        policy,
        max_tokens=max_tokens,
    ):
        if batch.error:
            result.errors.append(f"{batch.document_id}: {batch.error}")
            result.budget_exhausted = result.budget_exhausted or batch.budget_exhausted
            if on_progress:
                on_progress(batch.document_id, 0, batch.title, 0, batch.error)
            continue

        result.docs_processed += 1
        result.chunks_created += batch.chunks_created
        result.tokens_used += batch.tokens_used
        if on_progress:
            on_progress(
                batch.document_id,
                batch.chunks_created,
                batch.title,
                batch.tokens_used,
                None,
            )

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
    chunk_count, tokens_used, _ = await ingest_one(
        IngestionStore(session), doc, config, model, policy
    )
    return chunk_count, tokens_used


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
