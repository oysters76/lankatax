"""Semantic result cache: value types, ports and the Postgres store."""

from __future__ import annotations

from lankatax_ai.backend.cache.limits import DEFAULT_TTL_HOURS, SIMILARITY_THRESHOLD
from lankatax_ai.backend.cache.store import PgResultCache
from lankatax_ai.backend.cache.types import CacheAdmin, CacheStats, ResultCache

__all__ = [
    "DEFAULT_TTL_HOURS",
    "SIMILARITY_THRESHOLD",
    "CacheAdmin",
    "CacheStats",
    "PgResultCache",
    "ResultCache",
]
