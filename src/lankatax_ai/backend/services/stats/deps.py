"""Production wiring for the search-stats use case."""

from __future__ import annotations

from lankatax_ai.backend.analytics import NO_RESULT_THRESHOLD, SearchAnalytics
from lankatax_ai.backend.services.stats.types import StatsDeps


def build_stats_deps() -> StatsDeps:
    """Wire the production reader for the stats service."""
    return StatsDeps(
        reader_factory=SearchAnalytics,
        no_result_threshold=NO_RESULT_THRESHOLD,
    )
