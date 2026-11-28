"""Pipeline worker entrypoint."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from hatchet_sdk import Hatchet
from hatchet_sdk.worker.worker import Worker

from lankatax_ai.backend.config import settings
from lankatax_ai.backend.db.session import dispose_engine
from lankatax_ai.backend.jobs.client import build_hatchet_client
from lankatax_ai.backend.jobs.deps import JobRuntime, build_job_runtime
from lankatax_ai.backend.jobs.workflows.cache import build_cache_expire_workflow
from lankatax_ai.backend.jobs.workflows.ingest import (
    build_ingest_document_workflow,
    build_ingest_pending_workflow,
)
from lankatax_ai.backend.jobs.workflows.refresh import build_refresh_workflow
from lankatax_ai.backend.jobs.workflows.scrape import build_scrape_workflow

__all__ = ["build_pipeline_worker", "main"]


async def _lifespan() -> AsyncGenerator[None]:
    try:
        yield
    finally:
        await dispose_engine()


def build_pipeline_worker(hatchet: Hatchet, runtime: JobRuntime) -> Worker:
    """Register pipeline workflows on a worker."""
    return hatchet.worker(
        f"{settings.hatchet_worker_name}-pipeline",
        slots=settings.hatchet_worker_slots,
        workflows=[
            build_refresh_workflow(hatchet, runtime),
            build_scrape_workflow(hatchet, runtime),
            build_ingest_pending_workflow(hatchet, runtime),
            build_ingest_document_workflow(hatchet, runtime),
            build_cache_expire_workflow(hatchet, runtime),
        ],
        lifespan=_lifespan,
    )


def main() -> None:
    """Build and run the pipeline worker."""
    hatchet = build_hatchet_client()
    runtime = build_job_runtime()
    build_pipeline_worker(hatchet, runtime).start()


if __name__ == "__main__":
    main()
