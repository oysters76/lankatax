"""Hatchet workflow registration for a full corpus refresh."""

from __future__ import annotations

from hatchet_sdk import Hatchet
from hatchet_sdk.context.context import Context
from hatchet_sdk.runnables.workflow import Workflow

from lankatax_ai.backend.jobs.contracts import (
    CacheExpireInput,
    CacheOutput,
    IngestInput,
    IngestOutput,
    RefreshInput,
    ScrapeInput,
    ScrapeOutput,
)
from lankatax_ai.backend.jobs.deps import JobRuntime
from lankatax_ai.backend.jobs.policies import (
    CACHE_EXECUTION_TIMEOUT,
    CACHE_RETRIES,
    INGEST_EXECUTION_TIMEOUT,
    INGEST_RETRIES,
    SCHEDULE_TIMEOUT,
    SCRAPE_EXECUTION_TIMEOUT,
    SCRAPE_RETRIES,
)
from lankatax_ai.backend.jobs.tasks.cache import run_cache_expire
from lankatax_ai.backend.jobs.tasks.ingest import run_ingest
from lankatax_ai.backend.jobs.tasks.scrape import run_scrape
from lankatax_ai.backend.services.jobs.types import JobType

__all__ = ["build_refresh_workflow"]


def build_refresh_workflow(
    hatchet: Hatchet,
    runtime: JobRuntime,
) -> Workflow[RefreshInput]:
    workflow = hatchet.workflow(
        name=JobType.pipeline_refresh.value,
        input_validator=RefreshInput,
    )

    @workflow.task(
        name="scrape-registry",
        retries=SCRAPE_RETRIES,
        execution_timeout=SCRAPE_EXECUTION_TIMEOUT,
        schedule_timeout=SCHEDULE_TIMEOUT,
        backoff_factor=2.0,
        backoff_max_seconds=10,
    )
    async def scrape_registry(payload: RefreshInput, ctx: Context) -> ScrapeOutput:
        return await run_scrape(
            ScrapeInput(
                full=payload.full,
                discover_pdfs=payload.discover_pdfs,
                batch_size=payload.batch_size,
            ),
            ctx,
            runtime,
        )

    @workflow.task(
        name="ingest-pending",
        parents=[scrape_registry],
        retries=INGEST_RETRIES,
        execution_timeout=INGEST_EXECUTION_TIMEOUT,
        schedule_timeout=SCHEDULE_TIMEOUT,
        backoff_factor=2.0,
        backoff_max_seconds=10,
    )
    async def ingest_pending(payload: RefreshInput, ctx: Context) -> IngestOutput:
        return await run_ingest(
            IngestInput(
                target=payload.target,
                max_documents=payload.max_documents,
                max_tokens=payload.max_tokens,
                batch_size=payload.batch_size,
            ),
            ctx,
            runtime,
        )

    @workflow.task(
        name="expire-cache",
        parents=[ingest_pending],
        retries=CACHE_RETRIES,
        execution_timeout=CACHE_EXECUTION_TIMEOUT,
        schedule_timeout=SCHEDULE_TIMEOUT,
    )
    async def expire_cache(payload: RefreshInput, ctx: Context) -> CacheOutput:
        return await run_cache_expire(CacheExpireInput(), ctx, runtime)

    return workflow
