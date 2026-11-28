"""Production wiring shared by Hatchet workers."""

from __future__ import annotations

import dataclasses

from lankatax_ai.backend.db.session import async_session
from lankatax_ai.backend.llm import build_embeddings
from lankatax_ai.backend.services.cache import CacheDeps, build_cache_deps
from lankatax_ai.backend.services.ingest import IngestDeps, build_ingest_deps
from lankatax_ai.backend.services.protocols import SessionFactory
from lankatax_ai.backend.services.scrape import ScrapeDeps, build_scrape_deps

__all__ = ["JobRuntime", "build_job_runtime"]


@dataclasses.dataclass(frozen=True, slots=True)
class JobRuntime:
    """Collaborators the Hatchet tasks need, built once per worker process."""

    session_factory: SessionFactory
    scrape: ScrapeDeps
    ingest: IngestDeps
    cache: CacheDeps


def build_job_runtime() -> JobRuntime:
    """Build the production collaborators for workers."""
    embeddings = build_embeddings()
    return JobRuntime(
        session_factory=async_session,
        scrape=build_scrape_deps(),
        ingest=build_ingest_deps(embeddings),
        cache=build_cache_deps(),
    )
