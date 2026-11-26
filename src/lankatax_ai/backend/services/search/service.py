"""The search use case: cache, preprocess, retrieve, merge, rerank, context.

Transport-agnostic. The CLI and the API call :func:`search` with a session
factory and wired collaborators, and get a :class:`SearchOutcome` back.
Formatting stays in the caller.
"""

from __future__ import annotations

import time

from lankatax_ai.backend.analytics.types import SearchTiming
from lankatax_ai.backend.llm.embeddings import EMBEDDING_MODEL
from lankatax_ai.backend.rag.context_builder import ContextBuilder
from lankatax_ai.backend.rag.query_preprocessor import QueryPreprocessor
from lankatax_ai.backend.rag.types import RetrievalResult
from lankatax_ai.backend.services.errors import ValidationError
from lankatax_ai.backend.services.protocols import SessionFactory
from lankatax_ai.backend.services.search.types import (
    SearchDeps,
    SearchOptions,
    SearchOutcome,
)
from lankatax_ai.backend.utils import merge_result_sets


async def search(
    query: str,
    options: SearchOptions,
    *,
    session_factory: SessionFactory,
    deps: SearchDeps,
) -> SearchOutcome:
    """Run one search and return its results with timings.

    All clients and adapters arrive through ``deps``; this function owns only
    the use-case flow.
    """
    if not query.strip():
        raise ValidationError("Query must not be empty")

    started = time.perf_counter()
    processed = QueryPreprocessor().process(query)

    queries = [processed.expanded]
    if options.multi_query:
        expander = deps.expander_factory(options.llm_provider)
        queries.extend(await expander.expand(processed.expanded))

    query_embedding: list[float] | None = None
    results: list[RetrievalResult] = []
    cache_hit = False

    async with session_factory() as session:
        cached: list[RetrievalResult] | None = None
        if options.use_cache:
            query_embedding = await deps.embeddings.aembed_query(processed.expanded)
            cached = await deps.cache_factory(session).get(processed.expanded, query_embedding)

        if cached is not None:
            cache_hit = True
            results = cached
        else:
            retriever = deps.retriever_factory(session, options)
            result_sets = [
                await retriever.search(
                    text,
                    mode=options.mode,
                    top_k=options.top_k,
                    filters=options.filters,
                    include_embeddings=options.include_context,
                )
                for text in queries
            ]

            if options.multi_query:
                results = merge_result_sets(result_sets)[: options.top_k]
            else:
                results = result_sets[0]

        search_end = time.perf_counter()

    rerank_end: float | None = None
    if options.rerank and results and not cache_hit:
        reranker = deps.reranker_factory(options.llm_provider, options.rerank_threshold)
        results = await reranker.rerank(query, results, top_k=options.top_k)
        rerank_end = time.perf_counter()

    context: str | None = None
    if options.include_context and results:
        context = ContextBuilder(
            token_budget=options.effective_token_budget,
            dedup_threshold=options.dedup_threshold,
            embedding_model=EMBEDDING_MODEL,
        ).build(results)

    timings = SearchTiming()
    timings.record(search_start=started, search_end=search_end, rerank_end=rerank_end)

    async with session_factory() as session:
        if not cache_hit and options.use_cache and results and query_embedding is not None:
            await deps.cache_factory(session).put(processed.expanded, query_embedding, results)

        analytics = deps.analytics_factory(session)
        await analytics.log_search(
            query_text=query,
            expanded_query=processed.expanded,
            query_type=processed.query_type,
            search_mode=options.mode,
            top_score=results[0].score if results else None,
            result_count=len(results),
            timing=timings,
            llm_provider=options.llm_provider if rerank_end is not None else None,
            cache_hit=cache_hit,
            multi_query=options.multi_query,
            source=options.source,
        )
        await session.commit()

    return SearchOutcome(
        processed_query=processed,
        results=results,
        timings=timings,
        cache_hit=cache_hit,
        multi_query=options.multi_query,
        context=context,
    )
