"""Types for the ingest use case.

The result types, the embeddings slice and the repository port live with the
ingestion pipeline, because the pipeline and its callers share them. This module
only adds the request shape and the wiring the use case needs.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable
from enum import Enum
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from lankatax_ai.backend.ingestion.types import (
    ChunkConfig,
    DocumentEmbeddings,
    EmbeddingPolicy,
    IngestionRepository,
    IngestionResult,
    IngestionStatus,
    ProgressCallback,
)
from lankatax_ai.backend.services.errors import ValidationError

__all__ = [
    "ChunkConfig",
    "CorpusFinalizer",
    "DocumentEmbeddings",
    "EmbeddingPolicy",
    "IngestDeps",
    "IngestOptions",
    "IngestTarget",
    "IngestionRepository",
    "IngestionResult",
    "IngestionStatus",
    "ProgressCallback",
    "RepositoryFactory",
]


class IngestTarget(Enum):
    pending = "pending"
    all = "all"
    document = "document"


@dataclasses.dataclass(frozen=True, slots=True)
class IngestOptions:
    """Which documents to ingest and how to chunk them."""

    config: ChunkConfig = dataclasses.field(default_factory=ChunkConfig)
    target: IngestTarget = IngestTarget.pending
    document_id: UUID | None = None
    max_documents: int | None = None
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.target is IngestTarget.document and self.document_id is None:
            raise ValidationError("document_id is required when target is 'document'")
        if self.target is not IngestTarget.document and self.document_id is not None:
            raise ValidationError("document_id is only valid when target is 'document'")
        if self.max_documents is not None and self.max_documents < 1:
            raise ValidationError("max_documents must be at least 1")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValidationError("max_tokens must be at least 1")


RepositoryFactory = Callable[[AsyncSession], IngestionRepository]
CorpusFinalizer = Callable[[str | None], Awaitable[int]]


@dataclasses.dataclass(frozen=True, slots=True)
class IngestDeps:
    """Collaborators the ingest use case needs, wired by the caller."""

    repository_factory: RepositoryFactory
    embeddings: DocumentEmbeddings
    embedding_policy: EmbeddingPolicy
    corpus_finalizer: CorpusFinalizer
