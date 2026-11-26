"""Ingest use case: options, deps, ports and the ``ingest`` entry point."""

from __future__ import annotations

from lankatax_ai.backend.services.ingest.deps import build_ingest_deps
from lankatax_ai.backend.services.ingest.service import get_ingestion_status, ingest
from lankatax_ai.backend.services.ingest.types import (
    ChunkConfig,
    DocumentEmbeddings,
    IngestDeps,
    IngestionRepository,
    IngestionResult,
    IngestionStatus,
    IngestOptions,
    IngestTarget,
    ProgressCallback,
    RepositoryFactory,
)

__all__ = [
    "ChunkConfig",
    "DocumentEmbeddings",
    "IngestDeps",
    "IngestOptions",
    "IngestTarget",
    "IngestionRepository",
    "IngestionResult",
    "IngestionStatus",
    "ProgressCallback",
    "RepositoryFactory",
    "build_ingest_deps",
    "get_ingestion_status",
    "ingest",
]
