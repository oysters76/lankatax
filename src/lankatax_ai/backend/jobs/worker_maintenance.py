"""Maintenance worker entrypoint."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from hatchet_sdk import Hatchet
from hatchet_sdk.worker.worker import Worker

from lankatax_ai.backend.config import settings
from lankatax_ai.backend.db.session import dispose_engine
from lankatax_ai.backend.jobs.client import build_hatchet_client
from lankatax_ai.backend.jobs.deps import JobRuntime, build_job_runtime
from lankatax_ai.backend.jobs.workflows.cache import build_cache_expire_workflow

__all__ = ["build_maintenance_worker", "main"]


async def _lifespan() -> AsyncGenerator[None]:
    try:
        yield
    finally:
        await dispose_engine()


def build_maintenance_worker(hatchet: Hatchet, runtime: JobRuntime) -> Worker:
    """Register maintenance workflows on a worker."""
    return hatchet.worker(
        f"{settings.hatchet_worker_name}-maintenance",
        slots=1,
        workflows=[build_cache_expire_workflow(hatchet, runtime)],
        lifespan=_lifespan,
    )


def main() -> None:
    """Build and run the maintenance worker."""
    hatchet = build_hatchet_client()
    runtime = build_job_runtime()
    build_maintenance_worker(hatchet, runtime).start()


if __name__ == "__main__":
    main()
