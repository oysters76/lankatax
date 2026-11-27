"""The ingest use case: chunk, embed and store documents.

Transport-agnostic. The CLI and the API pass a session factory, wired
collaborators and an optional progress callback; formatting stays in the caller.
"""

from __future__ import annotations

from lankatax_ai.backend.ingestion.pipeline import ingest_documents, ingest_one
from lankatax_ai.backend.ingestion.types import IngestionResult, IngestionStatus
from lankatax_ai.backend.services.errors import ValidationError
from lankatax_ai.backend.services.ingest.types import (
    IngestDeps,
    IngestOptions,
    IngestTarget,
    ProgressCallback,
)
from lankatax_ai.backend.services.protocols import SessionFactory


async def ingest(
    options: IngestOptions,
    *,
    session_factory: SessionFactory,
    deps: IngestDeps,
    on_progress: ProgressCallback | None = None,
) -> IngestionResult:
    """Ingest the documents selected by ``options``.

    Per-document commits and failure isolation are handled by the pipeline; this
    function only chooses the document set and owns the session.
    """
    async with session_factory() as session:
        repo = deps.repository_factory(session)

        if options.target is IngestTarget.document:
            document_id = options.document_id
            if document_id is None:
                raise ValidationError("document_id is required when target is 'document'")
            doc = await repo.get_document(document_id)
            if doc is None:
                return IngestionResult(errors=[f"Document {document_id} not found"])
            chunk_count, tokens_used = await ingest_one(
                repo, doc, options.config, deps.embeddings, deps.embedding_policy
            )
            await repo.commit()
            return IngestionResult(
                docs_processed=1,
                chunks_created=chunk_count,
                tokens_used=tokens_used,
            )

        if options.target is IngestTarget.all:
            doc_ids = await repo.all_document_ids()
        else:
            doc_ids = await repo.pending_document_ids()

        return await ingest_documents(
            repo,
            doc_ids,
            options.config,
            deps.embeddings,
            deps.embedding_policy,
            on_progress,
        )


async def get_ingestion_status(
    *, session_factory: SessionFactory, deps: IngestDeps
) -> IngestionStatus:
    """Return how many documents are ingested versus pending."""
    async with session_factory() as session:
        return await deps.repository_factory(session).status()
