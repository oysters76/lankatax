from __future__ import annotations

import dataclasses
import logging

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from lankatax_ai.backend.db.models import SearchLog
from lankatax_ai.backend.rag.types import QueryType, SearchMode, SearchSource
from lankatax_ai.backend.types import LlmProvider

logger = logging.getLogger(__name__)

NO_RESULT_THRESHOLD = 0.3


@dataclasses.dataclass
class SearchTiming:
    embed_ms: int = 0
    search_ms: int = 0
    rerank_ms: int = 0
    total_ms: int = 0

    def record(
        self, *, search_start: float, search_end: float, rerank_end: float | None = None
    ) -> None:
        self.search_ms = int((search_end - search_start) * 1000)
        if rerank_end is not None:
            self.rerank_ms = int((rerank_end - search_end) * 1000)
        self.total_ms = self.search_ms + self.rerank_ms

    @property
    def elapsed_seconds(self) -> float:
        return self.total_ms / 1000


@dataclasses.dataclass
class SearchStats:
    total_queries: int
    avg_latency_ms: float
    hit_rate: float
    no_result_queries: list[str]
    avg_top_score: float
    cache_hit_count: int
    multi_query_count: int


class SearchAnalytics:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def log_search(
        self,
        *,
        query_text: str,
        expanded_query: str | None,
        query_type: QueryType,
        search_mode: SearchMode,
        top_score: float | None,
        result_count: int,
        timing: SearchTiming,
        llm_provider: LlmProvider | None = None,
        cache_hit: bool = False,
        multi_query: bool = False,
        source: SearchSource = SearchSource.cli,
    ) -> None:
        """Persist one search event to the search_logs table."""
        entry = SearchLog(
            query_text=query_text,
            expanded_query=expanded_query,
            query_type=query_type.value,
            search_mode=search_mode.value,
            top_score=top_score,
            result_count=result_count,
            latency_embed_ms=timing.embed_ms,
            latency_search_ms=timing.search_ms,
            latency_rerank_ms=timing.rerank_ms,
            latency_total_ms=timing.total_ms,
            llm_provider=llm_provider.value if llm_provider else None,
            cache_hit=cache_hit,
            multi_query=multi_query,
            source=source.value,
        )
        self._session.add(entry)
        logger.debug(
            "Logged search: query=%r score=%.3f latency=%dms",
            query_text,
            top_score or 0,
            timing.total_ms,
        )

    async def get_stats(self) -> SearchStats:
        """Compute aggregate search quality metrics."""
        # total queries and averages
        summary = (
            (
                await self._session.execute(
                    select(
                        func.count().label("total"),
                        func.avg(SearchLog.latency_total_ms).label("avg_latency"),
                        func.avg(SearchLog.top_score).label("avg_score"),
                        func.count().filter(SearchLog.cache_hit.is_(True)).label("cache_hits"),
                        func.count().filter(SearchLog.multi_query.is_(True)).label("multi_queries"),
                    )
                )
            )
            .mappings()
            .one()
        )

        total = summary["total"]
        if total == 0:
            return SearchStats(
                total_queries=0,
                avg_latency_ms=0.0,
                hit_rate=0.0,
                no_result_queries=[],
                avg_top_score=0.0,
                cache_hit_count=0,
                multi_query_count=0,
            )

        # no-result queries: top_score below threshold or zero results
        no_result_rows = (
            (
                await self._session.execute(
                    select(SearchLog.query_text)
                    .where(
                        (SearchLog.top_score < NO_RESULT_THRESHOLD) | (SearchLog.result_count == 0)
                    )
                    .order_by(SearchLog.created_at.desc())
                    .limit(20)
                )
            )
            .scalars()
            .all()
        )

        # hit rate: queries that returned meaningful results
        hit_count = (
            await self._session.execute(
                select(func.count()).where(
                    SearchLog.top_score >= NO_RESULT_THRESHOLD,
                    SearchLog.result_count > 0,
                )
            )
        ).scalar()

        return SearchStats(
            total_queries=total,
            avg_latency_ms=float(summary["avg_latency"] or 0),
            hit_rate=hit_count / total if total else 0.0,
            no_result_queries=list(no_result_rows),
            avg_top_score=float(summary["avg_score"] or 0),
            cache_hit_count=summary["cache_hits"],
            multi_query_count=summary["multi_queries"],
        )
