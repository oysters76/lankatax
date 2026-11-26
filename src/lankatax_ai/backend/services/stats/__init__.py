"""Search-stats use case: deps, result type and the ``get_stats`` entry point."""

from __future__ import annotations

from lankatax_ai.backend.services.stats.deps import build_stats_deps
from lankatax_ai.backend.services.stats.service import get_stats
from lankatax_ai.backend.services.stats.types import (
    SearchStats,
    StatsDeps,
    StatsReader,
    StatsReaderFactory,
    StatsSnapshot,
)

__all__ = [
    "SearchStats",
    "StatsDeps",
    "StatsReader",
    "StatsReaderFactory",
    "StatsSnapshot",
    "build_stats_deps",
    "get_stats",
]
