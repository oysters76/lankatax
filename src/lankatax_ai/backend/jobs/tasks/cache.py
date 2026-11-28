"""Cache maintenance task execution."""

from __future__ import annotations

import time

from hatchet_sdk.context.context import Context

from lankatax_ai.backend.jobs.contracts import CacheExpireInput, CacheOutput
from lankatax_ai.backend.jobs.deps import JobRuntime
from lankatax_ai.backend.services.cache import invalidate_expired

__all__ = ["run_cache_expire"]


async def run_cache_expire(
    payload: CacheExpireInput,
    ctx: Context,
    runtime: JobRuntime,
) -> CacheOutput:
    """Expire TTL and previous-epoch cache entries."""
    started = time.perf_counter()
    expired = await invalidate_expired(
        session_factory=runtime.session_factory,
        deps=runtime.cache,
    )
    ctx.log({"event": "cache.expired", "expired": expired})
    return CacheOutput(
        expired=expired,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )
