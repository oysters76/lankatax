from __future__ import annotations

import dataclasses

import pytest

from lankatax_ai.backend.analytics.types import SearchTiming
from lankatax_ai.backend.rag.types import PreprocessedQuery
from lankatax_ai.backend.services.errors import (
    NotFoundError,
    ServiceError,
    UpstreamError,
    ValidationError,
)
from lankatax_ai.backend.services.search import SearchOptions, SearchOutcome, limits
from lankatax_ai.backend.types import (
    MODEL_TOKEN_BUDGETS,
    LlmProvider,
    QueryType,
    SearchMode,
    SearchSource,
)

pytestmark = pytest.mark.unit


def _options(**overrides) -> SearchOptions:
    return SearchOptions(source=SearchSource.api, **overrides)


def test_defaults_mirror_cli_behaviour():
    options = _options()

    assert options.mode is SearchMode.hybrid
    assert options.top_k == 5
    assert options.rerank is False
    assert options.multi_query is False
    assert options.use_cache is True
    assert options.include_context is False
    assert options.vector_weight == 0.7
    assert options.rerank_threshold == 3
    assert options.dedup_threshold == 0.6
    assert options.llm_provider is LlmProvider.gemini
    assert options.filters.has_any is False


def test_effective_token_budget_falls_back_to_provider():
    for provider, budget in MODEL_TOKEN_BUDGETS.items():
        assert _options(llm_provider=provider).effective_token_budget == budget


def test_explicit_token_budget_wins():
    assert _options(token_budget=1200).effective_token_budget == 1200


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("top_k", 0, "Top-k must be 1-50"),
        ("top_k", 51, "Top-k must be 1-50"),
        ("vector_weight", -0.1, "Vector weight must be 0.0-1.0"),
        ("vector_weight", 1.1, "Vector weight must be 0.0-1.0"),
        ("rerank_threshold", 0, "Rerank threshold must be 1-10"),
        ("rerank_threshold", 11, "Rerank threshold must be 1-10"),
        ("dedup_threshold", -0.1, "Dedup threshold must be 0.0-1.0"),
        ("dedup_threshold", 1.1, "Dedup threshold must be 0.0-1.0"),
        ("token_budget", 499, "Token budget must be 500-16000"),
        ("token_budget", 16001, "Token budget must be 500-16000"),
    ],
)
def test_invalid_options_raise_validation_error(field, value, message):
    with pytest.raises(ValidationError, match=message):
        _options(**{field: value})


def test_boundary_values_are_accepted():
    assert _options(top_k=limits.MIN_TOP_K).top_k == limits.MIN_TOP_K
    assert _options(top_k=limits.MAX_TOP_K).top_k == limits.MAX_TOP_K
    assert (
        _options(token_budget=limits.MIN_TOKEN_BUDGET).effective_token_budget
        == limits.MIN_TOKEN_BUDGET
    )
    assert (
        _options(token_budget=limits.MAX_TOKEN_BUDGET).effective_token_budget
        == limits.MAX_TOKEN_BUDGET
    )


def test_options_are_immutable():
    options = _options()

    with pytest.raises(dataclasses.FrozenInstanceError):
        options.top_k = 10  # type: ignore[misc]


def test_search_outcome_defaults():
    outcome = SearchOutcome(
        processed_query=PreprocessedQuery(
            original="apit", expanded="apit", query_type=QueryType.factual
        ),
        results=[],
        timings=SearchTiming(),
    )

    assert outcome.cache_hit is False
    assert outcome.multi_query is False
    assert outcome.context is None
    assert outcome.has_results is False


def test_models_are_slot_based():
    assert not hasattr(_options(), "__dict__")

    outcome = SearchOutcome(
        processed_query=PreprocessedQuery(
            original="apit", expanded="apit", query_type=QueryType.factual
        ),
        results=[],
        timings=SearchTiming(),
    )
    assert not hasattr(outcome, "__dict__")
    with pytest.raises(AttributeError):
        outcome.unknown = "typo"  # type: ignore[attr-defined]


def test_error_taxonomy():
    assert issubclass(ValidationError, ServiceError)
    assert issubclass(NotFoundError, ServiceError)
    assert issubclass(UpstreamError, ServiceError)
    assert ServiceError.code == "service_error"
    assert ValidationError.code == "validation_error"
    assert NotFoundError.code == "not_found"
    assert UpstreamError.code == "upstream_error"
