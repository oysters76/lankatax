"""Request, result and collaborator types for the search use case.

These types carry no transport concerns: no Rich renderables, no Typer
parameters, no HTTP schemas. The CLI and the API both translate into and out of
them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from lankatax_ai.backend.analytics.types import SearchLogWriter, SearchTiming
from lankatax_ai.backend.cache.types import ResultCache
from lankatax_ai.backend.rag.metadata_filter import MetadataFilters
from lankatax_ai.backend.rag.types import (
    PreprocessedQuery,
    RetrievalResult,
)
from lankatax_ai.backend.services.protocols import Embeddings
from lankatax_ai.backend.services.search import limits
from lankatax_ai.backend.services.utils import require_range
from lankatax_ai.backend.types import (
    MODEL_TOKEN_BUDGETS,
    LlmProvider,
    SearchMode,
    SearchSource,
)


class Retriever(Protocol):
    async def search(
        self,
        query: str,
        *,
        mode: SearchMode,
        top_k: int,
        filters: MetadataFilters | None,
        include_embeddings: bool,
    ) -> list[RetrievalResult]: ...


class Expander(Protocol):
    async def expand(self, query: str) -> list[str]: ...


class RerankLike(Protocol):
    async def rerank(
        self, query: str, results: list[RetrievalResult], *, top_k: int
    ) -> list[RetrievalResult]: ...


@dataclasses.dataclass(frozen=True, slots=True)
class SearchOptions:
    """Everything a search run needs besides the query and a session.

    Defaults mirror the CLI defaults so existing behaviour is preserved. Values
    are validated here rather than in the caller, because every transport needs
    the same rules.
    """

    source: SearchSource
    mode: SearchMode = SearchMode.hybrid
    filters: MetadataFilters = dataclasses.field(default_factory=MetadataFilters)
    top_k: int = limits.DEFAULT_TOP_K
    rerank: bool = False
    multi_query: bool = False
    use_cache: bool = True
    include_context: bool = False
    vector_weight: float = limits.DEFAULT_VECTOR_WEIGHT
    rerank_threshold: int = limits.DEFAULT_RERANK_THRESHOLD
    dedup_threshold: float = limits.DEFAULT_DEDUP_THRESHOLD
    token_budget: int | None = None
    llm_provider: LlmProvider = LlmProvider.gemini

    def __post_init__(self) -> None:
        require_range("Top-k", self.top_k, limits.MIN_TOP_K, limits.MAX_TOP_K)
        require_range(
            "Vector weight", self.vector_weight, limits.MIN_VECTOR_WEIGHT, limits.MAX_VECTOR_WEIGHT
        )
        require_range(
            "Rerank threshold",
            self.rerank_threshold,
            limits.MIN_RERANK_THRESHOLD,
            limits.MAX_RERANK_THRESHOLD,
        )
        require_range(
            "Dedup threshold",
            self.dedup_threshold,
            limits.MIN_DEDUP_THRESHOLD,
            limits.MAX_DEDUP_THRESHOLD,
        )
        if self.token_budget is not None:
            require_range(
                "Token budget", self.token_budget, limits.MIN_TOKEN_BUDGET, limits.MAX_TOKEN_BUDGET
            )

    @property
    def effective_token_budget(self) -> int:
        """Explicit budget when set, otherwise the provider's default."""
        if self.token_budget is not None:
            return self.token_budget
        return MODEL_TOKEN_BUDGETS[self.llm_provider]


@dataclasses.dataclass(slots=True)
class SearchOutcome:
    """Result of one search run, independent of how it will be displayed."""

    processed_query: PreprocessedQuery
    results: list[RetrievalResult]
    timings: SearchTiming
    cache_hit: bool = False
    multi_query: bool = False
    context: str | None = None

    @property
    def has_results(self) -> bool:
        return bool(self.results)


RetrieverFactory = Callable[[AsyncSession, SearchOptions], Retriever]
ExpanderFactory = Callable[[LlmProvider], Expander]
RerankerFactory = Callable[[LlmProvider, int], RerankLike]
CacheFactory = Callable[[AsyncSession], ResultCache]
AnalyticsFactory = Callable[[AsyncSession], SearchLogWriter]


@dataclasses.dataclass(frozen=True, slots=True)
class SearchDeps:
    """Collaborators the search use case needs, wired by the caller.

    The service never builds clients or adapters itself. ``build_search_deps``
    provides the production wiring; tests pass fakes.
    """

    embeddings: Embeddings
    retriever_factory: RetrieverFactory
    expander_factory: ExpanderFactory
    reranker_factory: RerankerFactory
    cache_factory: CacheFactory
    analytics_factory: AnalyticsFactory
