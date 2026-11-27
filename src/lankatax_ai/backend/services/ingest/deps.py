"""Production wiring for the ingest use case.

The embeddings client is built by the transport and passed in, so the service
never constructs an I/O client itself.
"""

from __future__ import annotations

from lankatax_ai.backend.db.session import async_session
from lankatax_ai.backend.ingestion.policy import build_embedding_policy
from lankatax_ai.backend.ingestion.store import IngestionStore
from lankatax_ai.backend.ingestion.types import DocumentEmbeddings
from lankatax_ai.backend.services.corpus import finalize_ingest
from lankatax_ai.backend.services.ingest.types import IngestDeps


async def _finalize_corpus(run_id: str | None) -> int:
    return await finalize_ingest(session_factory=async_session, run_id=run_id)


def build_ingest_deps(embeddings: DocumentEmbeddings) -> IngestDeps:
    """Wire the production repository, embeddings, policy and finalizer."""
    return IngestDeps(
        repository_factory=IngestionStore,
        embeddings=embeddings,
        embedding_policy=build_embedding_policy(),
        corpus_finalizer=_finalize_corpus,
    )
