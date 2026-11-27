"""Production wiring for the ingest use case.

The embeddings client is built by the transport and passed in, so the service
never constructs an I/O client itself.
"""

from __future__ import annotations

from lankatax_ai.backend.ingestion.policy import build_embedding_policy
from lankatax_ai.backend.ingestion.store import IngestionStore
from lankatax_ai.backend.ingestion.types import DocumentEmbeddings
from lankatax_ai.backend.services.ingest.types import IngestDeps


def build_ingest_deps(embeddings: DocumentEmbeddings) -> IngestDeps:
    """Wire the production repository, embeddings and policy."""
    return IngestDeps(
        repository_factory=IngestionStore,
        embeddings=embeddings,
        embedding_policy=build_embedding_policy(),
    )
