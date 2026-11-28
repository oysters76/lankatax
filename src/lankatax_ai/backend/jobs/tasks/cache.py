"""Cache maintenance task execution."""

from __future__ import annotations

import time

from hatchet_sdk.context.context import Context

from lankatax_ai.backend.jobs.contracts import CacheExpireInput, CacheOutput
from lankatax_ai.backend.jobs.deps import JobRuntime
from lankatax_ai.backend.jobs.observability import (
    job_event,
    record_job_completed,
    record_job_started,
)
from lankatax_ai.backend.services.cache import invalidate_expired
from lankatax_ai.backend.services.jobs.types import JobType

__all__ = ["run_cache_expire"]


async def run_cache_expire(
    payload: CacheExpireInput,
    ctx: Context,
    runtime: JobRuntime,
) -> CacheOutput:
    """Expire TTL and previous-epoch cache entries."""
    started = time.perf_counter()
    record_job_started(JobType.cache_expire)
    expired = await invalidate_expired(
        session_factory=runtime.session_factory,
        deps=runtime.cache,
    )
    duration_ms = int((time.perf_counter() - started) * 1000)
    job_event(
        ctx,
        event="cache.expired",
        job_type=JobType.cache_expire,
        expired=expired,
    )
    record_job_completed(
        JobType.cache_expire,
        status="succeeded",
        duration_ms=duration_ms,
    )
    return CacheOutput(expired=expired, duration_ms=duration_ms)
