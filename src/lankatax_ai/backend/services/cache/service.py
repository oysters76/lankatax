"""The cache use cases: report statistics, clear entries, expire old ones.

Transport-agnostic. Each function owns its session and commits its own writes,
so callers only supply a session factory and wired collaborators.
"""

from __future__ import annotations

from lankatax_ai.backend.cache.types import CacheStats
from lankatax_ai.backend.services.cache.types import CacheDeps
from lankatax_ai.backend.services.protocols import SessionFactory


async def get_cache_stats(*, session_factory: SessionFactory, deps: CacheDeps) -> CacheStats:
    """Return aggregate cache metrics."""
    async with session_factory() as session:
        return await deps.admin_factory(session).get_stats()


async def clear_cache(*, session_factory: SessionFactory, deps: CacheDeps) -> int:
    """Delete every cache entry and return how many were removed."""
    async with session_factory() as session:
        deleted = await deps.admin_factory(session).clear()
        await session.commit()
        return deleted


async def invalidate_expired(*, session_factory: SessionFactory, deps: CacheDeps) -> int:
    """Remove entries past their TTL and return how many were removed."""
    async with session_factory() as session:
        removed = await deps.admin_factory(session).invalidate_expired()
        await session.commit()
        return removed
