"""The search-stats use case: read aggregates and derive quality metrics."""

from __future__ import annotations

from lankatax_ai.backend.services.protocols import SessionFactory
from lankatax_ai.backend.services.stats.types import SearchStats, StatsDeps


async def get_stats(*, session_factory: SessionFactory, deps: StatsDeps) -> SearchStats:
    """Return aggregate search metrics.

    The reader owns the SQL; this use case owns the policy (threshold and the
    empty-dataset contract) so every transport reports the same numbers.
    """
    async with session_factory() as session:
        snapshot = await deps.reader_factory(session).snapshot(threshold=deps.no_result_threshold)

    total = snapshot.total_queries
    return SearchStats(
        total_queries=total,
        avg_latency_ms=snapshot.avg_latency_ms,
        hit_rate=snapshot.hit_count / total if total else 0.0,
        no_result_queries=list(snapshot.no_result_queries),
        avg_top_score=snapshot.avg_top_score,
        cache_hit_count=snapshot.cache_hit_count,
        multi_query_count=snapshot.multi_query_count,
    )
