"""Hatchet workflow registration for cache maintenance."""

from __future__ import annotations

from hatchet_sdk import Hatchet
from hatchet_sdk.context.context import Context
from hatchet_sdk.runnables.workflow import Workflow

from lankatax_ai.backend.jobs.contracts import CacheExpireInput, CacheOutput
from lankatax_ai.backend.jobs.deps import JobRuntime
from lankatax_ai.backend.jobs.policies import (
    CACHE_EXECUTION_TIMEOUT,
    CACHE_RETRIES,
    SCHEDULE_TIMEOUT,
)
from lankatax_ai.backend.jobs.tasks.cache import run_cache_expire
from lankatax_ai.backend.services.jobs.types import JobType

__all__ = ["build_cache_expire_workflow"]


def build_cache_expire_workflow(
    hatchet: Hatchet,
    runtime: JobRuntime,
) -> Workflow[CacheExpireInput]:
    workflow = hatchet.workflow(
        name=JobType.cache_expire.value,
        input_validator=CacheExpireInput,
    )

    @workflow.task(
        name="expire-cache",
        retries=CACHE_RETRIES,
        execution_timeout=CACHE_EXECUTION_TIMEOUT,
        schedule_timeout=SCHEDULE_TIMEOUT,
    )
    async def expire_cache(payload: CacheExpireInput, ctx: Context) -> CacheOutput:
        return await run_cache_expire(payload, ctx, runtime)

    return workflow
