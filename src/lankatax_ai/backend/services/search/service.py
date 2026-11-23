"""The search use case: preprocess, retrieve, merge, rerank, assemble context.

Transport-agnostic. The CLI and the API call :func:`search` with a session
factory and get a :class:`SearchOutcome` back. Formatting stays in the caller.
"""

from __future__ import annotations

import time
from typing import cast

from langchain_openai import OpenAIEmbeddings

from lankatax_ai.backend.config import settings
from lankatax_ai.backend.llm import build_llm
from lankatax_ai.backend.rag.context_builder import ContextBuilder
from lankatax_ai.backend.rag.multi_query_retriever import MultiQueryExpander
from lankatax_ai.backend.rag.query_preprocessor import QueryPreprocessor
from lankatax_ai.backend.rag.reranker import Reranker
from lankatax_ai.backend.rag.retriever import HybridRetriever
from lankatax_ai.backend.rag.search_analytics import SearchTiming
from lankatax_ai.backend.services.errors import ValidationError
from lankatax_ai.backend.services.protocols import Embeddings, SessionFactory
from lankatax_ai.backend.services.search.types import (
    Expander,
    RerankLike,
    Retriever,
    RetrieverFactory,
    SearchOptions,
    SearchOutcome,
)
from lankatax_ai.backend.utils import merge_result_sets

EMBEDDING_MODEL = "text-embedding-3-small"


def _default_embeddings() -> Embeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=settings.openai_api_key,
    )


async def search(
    query: str,
    options: SearchOptions,
    *,
    session_factory: SessionFactory,
    embeddings: Embeddings | None = None,
    retriever_factory: RetrieverFactory | None = None,
    expander: Expander | None = None,
    reranker: RerankLike | None = None,
) -> SearchOutcome:
    """Run one search and return its results with timings.

    ``embeddings``, ``retriever_factory``, ``expander`` and ``reranker`` exist so
    tests and the orchestrator can substitute components; production callers
    leave them unset.
    """
    if not query.strip():
        raise ValidationError("Query must not be empty")

    started = time.perf_counter()
    processed = QueryPreprocessor().process(query)

    queries = [processed.expanded]
    if options.multi_query:
        active_expander = expander or MultiQueryExpander(build_llm(options.llm_provider))
        queries.extend(await active_expander.expand(processed.expanded))

    async with session_factory() as session:
        retriever: Retriever
        if retriever_factory is not None:
            retriever = retriever_factory(session)
        else:
            embed_model = embeddings or _default_embeddings()
            retriever = HybridRetriever(
                session,
                cast(OpenAIEmbeddings, embed_model),
                vector_weight=options.vector_weight,
            )

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
    if options.rerank and results:
        active_reranker = reranker or Reranker(
            build_llm(options.llm_provider), threshold=options.rerank_threshold
        )
        results = await active_reranker.rerank(query, results, top_k=options.top_k)
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

    return SearchOutcome(
        processed_query=processed,
        results=results,
        timings=timings,
        cache_hit=False,
        multi_query=options.multi_query,
        context=context,
    )
