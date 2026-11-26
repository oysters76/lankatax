"""Search analytics: the search-event log, its value types and its ports."""

from __future__ import annotations

from lankatax_ai.backend.analytics.limits import NO_RESULT_THRESHOLD
from lankatax_ai.backend.analytics.store import SearchAnalytics
from lankatax_ai.backend.analytics.types import (
    SearchLogWriter,
    SearchStats,
    SearchTiming,
    StatsReader,
    StatsSnapshot,
)

__all__ = [
    "NO_RESULT_THRESHOLD",
    "SearchAnalytics",
    "SearchLogWriter",
    "SearchStats",
    "SearchTiming",
    "StatsReader",
    "StatsSnapshot",
]
