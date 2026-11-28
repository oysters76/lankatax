"""Production wiring for the ask use case.

The LangGraph adapter is built per option set and cached; callers pass the
resulting ``AskDeps`` to :func:`lankatax_ai.backend.services.ask.service.ask`.
"""

from __future__ import annotations

from typing import cast

from langchain_core.language_models import BaseChatModel

from lankatax_ai.backend.chat import GraphConfig, GraphDeps, build_ask_graph
from lankatax_ai.backend.config import settings
from lankatax_ai.backend.db.session import async_session
from lankatax_ai.backend.guardrails import (
    InputValidator,
    OutputChecker,
    QuotaLimiter,
    RateLimitRule,
)
from lankatax_ai.backend.llm import build_embeddings, build_llm
from lankatax_ai.backend.memory import SqlConversationStore
from lankatax_ai.backend.services.ask.types import (
    AnswerGraph,
    AskDeps,
    AskOptions,
)
from lankatax_ai.backend.services.protocols import Embeddings
from lankatax_ai.backend.services.search import (
    SearchDeps,
    SearchOptions,
    SearchOutcome,
    build_search_deps,
    search,
)
from lankatax_ai.backend.types import SearchSource

__all__ = ["build_ask_deps"]


def build_ask_deps(
    *,
    embeddings: Embeddings | None = None,
    search_deps: SearchDeps | None = None,
) -> AskDeps:
    """Wire the production store, limiter and graph factory.

    Callers that already built embeddings (the API) pass them in so the process
    keeps one embedding client and one search wiring.
    """
    if embeddings is None:
        embeddings = build_embeddings()
    if search_deps is None:
        search_deps = build_search_deps(embeddings)
    graphs: dict[AskOptions, AnswerGraph] = {}

    def graph_factory(options: AskOptions) -> AnswerGraph:
        graph = graphs.get(options)
        if graph is None:
            graph = _build_graph(options, search_deps)
            graphs[options] = graph
        return graph

    return AskDeps(
        graph_factory=graph_factory,
        conversation_store=SqlConversationStore(async_session),
        rate_limiter=QuotaLimiter(
            [
                RateLimitRule(settings.ask_rate_limit_per_minute, 60.0),
                RateLimitRule(settings.ask_rate_limit_per_day, 86_400.0),
            ],
            global_rules=[
                RateLimitRule(settings.ask_global_rate_limit_per_minute, 60.0),
                RateLimitRule(settings.ask_global_rate_limit_per_day, 86_400.0),
            ],
        ),
    )


def _build_graph(options: AskOptions, search_deps: SearchDeps) -> AnswerGraph:
    llm: BaseChatModel = build_llm(options.provider, max_tokens=options.answer_max_tokens)

    async def run_search(query: str) -> SearchOutcome:
        return await search(
            query,
            _search_options(options),
            session_factory=async_session,
            deps=search_deps,
        )

    config = GraphConfig(
        min_score=options.min_score,
        guard_input=options.guard_input,
        guard_output=options.guard_output,
    )
    graph = build_ask_graph(
        GraphDeps(
            llm=llm,
            search=run_search,
            config=config,
            input_validator=(
                InputValidator(llm=llm, max_chars=settings.ask_max_question_chars)
                if options.guard_input
                else None
            ),
            output_checker=OutputChecker(llm=llm) if options.guard_output else None,
        )
    )
    return cast(AnswerGraph, graph)


def _search_options(options: AskOptions) -> SearchOptions:
    return SearchOptions(
        source=SearchSource.api,
        mode=options.mode,
        top_k=options.top_k,
        rerank=options.rerank,
        include_context=True,
        token_budget=options.context_token_budget,
        llm_provider=options.provider,
    )
