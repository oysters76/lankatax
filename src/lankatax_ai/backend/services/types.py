"""Plain request and result models exchanged with the service layer.

These types carry no transport concerns: no Rich renderables, no Typer
parameters, no HTTP schemas. The CLI and the API both translate into and out of
them.
"""

from __future__ import annotations

import dataclasses

from lankatax_ai.backend.rag.metadata_filter import MetadataFilters
from lankatax_ai.backend.rag.search_analytics import SearchTiming
from lankatax_ai.backend.rag.types import (
    PreprocessedQuery,
    RetrievalResult,
    SearchMode,
    SearchSource,
)
from lankatax_ai.backend.services import limits
from lankatax_ai.backend.services.utils import require_range
from lankatax_ai.backend.types import MODEL_TOKEN_BUDGETS, LlmProvider


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
