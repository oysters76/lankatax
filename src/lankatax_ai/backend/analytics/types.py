"""Value types and ports for the search-event log.

The search use case records events; the stats use case reports on them. Both
talk to the store through the small protocols below, so neither depends on the
SQL adapter.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol

from lankatax_ai.backend.types import LlmProvider, QueryType, SearchMode, SearchSource


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


@dataclasses.dataclass(frozen=True, slots=True)
class StatsSnapshot:
    """Raw aggregates read from ``search_logs``, before metrics are derived."""

    total_queries: int
    avg_latency_ms: float
    avg_top_score: float
    cache_hit_count: int
    multi_query_count: int
    hit_count: int
    no_result_queries: list[str]


class SearchLogWriter(Protocol):
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
        source: SearchSource,
        llm_provider: LlmProvider | None = None,
        cache_hit: bool = False,
        multi_query: bool = False,
    ) -> None: ...


class StatsReader(Protocol):
    async def snapshot(self, *, threshold: float) -> StatsSnapshot: ...
