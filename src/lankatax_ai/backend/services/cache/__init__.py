"""Cache use case: deps, admin port and the cache entry points."""

from __future__ import annotations

from lankatax_ai.backend.services.cache.deps import build_cache_deps
from lankatax_ai.backend.services.cache.service import (
    clear_cache,
    get_cache_stats,
    invalidate_expired,
)
from lankatax_ai.backend.services.cache.types import (
    CacheAdmin,
    CacheAdminFactory,
    CacheDeps,
    CacheStats,
)

__all__ = [
    "CacheAdmin",
    "CacheAdminFactory",
    "CacheDeps",
    "CacheStats",
    "build_cache_deps",
    "clear_cache",
    "get_cache_stats",
    "invalidate_expired",
]
