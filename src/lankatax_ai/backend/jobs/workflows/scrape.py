"""Hatchet workflow registration for registry scraping."""

from __future__ import annotations

from hatchet_sdk import Hatchet
from hatchet_sdk.context.context import Context
from hatchet_sdk.runnables.workflow import Workflow

from lankatax_ai.backend.jobs.contracts import ScrapeInput, ScrapeOutput
from lankatax_ai.backend.jobs.deps import JobRuntime
from lankatax_ai.backend.jobs.policies import (
    SCHEDULE_TIMEOUT,
    SCRAPE_EXECUTION_TIMEOUT,
    SCRAPE_RETRIES,
)
from lankatax_ai.backend.jobs.tasks.scrape import run_scrape
from lankatax_ai.backend.services.jobs.types import JobType

__all__ = ["build_scrape_workflow"]


def build_scrape_workflow(hatchet: Hatchet, runtime: JobRuntime) -> Workflow[ScrapeInput]:
    """Build the standalone scrape workflow."""
    workflow = hatchet.workflow(
        name=JobType.scrape_registry.value,
        input_validator=ScrapeInput,
    )

    @workflow.task(
        name="scrape-registry",
        retries=SCRAPE_RETRIES,
        execution_timeout=SCRAPE_EXECUTION_TIMEOUT,
        schedule_timeout=SCHEDULE_TIMEOUT,
        backoff_factor=2.0,
        backoff_max_seconds=10,
    )
    async def scrape_registry(payload: ScrapeInput, ctx: Context) -> ScrapeOutput:
        return await run_scrape(payload, ctx, runtime)

    return workflow
