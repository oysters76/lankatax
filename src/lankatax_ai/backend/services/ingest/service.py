"""The ingest use case: chunk, embed and store documents.

Transport-agnostic. Callers pass a session factory and wired collaborators;
formatting and progress stay outside the use case.
"""

from __future__ import annotations

from lankatax_ai.backend.ingestion.pipeline import ingest_batches
from lankatax_ai.backend.ingestion.types import IngestionResult, IngestionStatus
from lankatax_ai.backend.services.ingest.types import (
    IngestDeps,
    IngestOptions,
    IngestTarget,
)
from lankatax_ai.backend.services.protocols import SessionFactory


async def ingest(
    options: IngestOptions,
    *,
    session_factory: SessionFactory,
    deps: IngestDeps,
    run_id: str | None = None,
) -> IngestionResult:
    """Ingest the documents selected by ``options``.

    Per-document commits and failure isolation are handled by the pipeline; this
    function only chooses the document set and owns the session.
    """
    async with session_factory() as session:
        repo = deps.repository_factory(session)

        if options.target is IngestTarget.document:
            document_id = options.document_id
            assert document_id is not None  # IngestOptions validates this
            doc_ids = [document_id]
        elif options.target is IngestTarget.all:
            doc_ids = await repo.all_document_ids()
        else:
            doc_ids = await repo.pending_document_ids()

        if options.max_documents is not None:
            doc_ids = doc_ids[: options.max_documents]

        result = IngestionResult()
        async for batch in ingest_batches(
            repo,
            doc_ids,
            options.config,
            deps.embeddings,
            deps.embedding_policy,
            max_tokens=options.max_tokens,
        ):
            if batch.error:
                result.errors.append(f"{batch.document_id}: {batch.error}")
                result.budget_exhausted = result.budget_exhausted or batch.budget_exhausted
                continue
            result.docs_processed += 1
            result.chunks_created += batch.chunks_created
            result.tokens_used += batch.tokens_used

    await deps.corpus_finalizer(run_id)
    return result


async def get_ingestion_status(
    *, session_factory: SessionFactory, deps: IngestDeps
) -> IngestionStatus:
    """Return how many documents are ingested versus pending."""
    async with session_factory() as session:
        return await deps.repository_factory(session).status()
