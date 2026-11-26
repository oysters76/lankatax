"""Production wiring for the cache use case."""

from __future__ import annotations

from lankatax_ai.backend.cache import PgResultCache
from lankatax_ai.backend.services.cache.types import CacheDeps


def build_cache_deps() -> CacheDeps:
    """Wire the production cache store for the cache service."""
    return CacheDeps(admin_factory=PgResultCache)
