"""Production wiring for the search use case.

Transports and the orchestrator call :func:`build_search_deps` once and pass the
result to :func:`lankatax_ai.backend.services.search.service.search`. The service
itself never constructs clients or adapters.
"""

from __future__ import annotations

from typing import cast

from langchain_openai import OpenAIEmbeddings
from sqlalchemy.ext.asyncio import AsyncSession

from lankatax_ai.backend.analytics import SearchAnalytics
from lankatax_ai.backend.llm import build_llm
from lankatax_ai.backend.rag.multi_query_retriever import MultiQueryExpander
from lankatax_ai.backend.rag.reranker import Reranker
from lankatax_ai.backend.rag.result_cache import PgResultCache
from lankatax_ai.backend.rag.retriever import HybridRetriever
from lankatax_ai.backend.services.protocols import Embeddings
from lankatax_ai.backend.services.search.types import SearchDeps, SearchOptions


def build_search_deps(embeddings: Embeddings) -> SearchDeps:
    """Wire the production collaborators for the search service.

    The embeddings client is shared across requests. Expanders and rerankers are
    built on demand because the provider and threshold come from each request's
    options.
    """

    def retriever_factory(session: AsyncSession, options: SearchOptions) -> HybridRetriever:
        return HybridRetriever(
            session,
            cast(OpenAIEmbeddings, embeddings),
            vector_weight=options.vector_weight,
        )

    return SearchDeps(
        embeddings=embeddings,
        retriever_factory=retriever_factory,
        expander_factory=lambda provider: MultiQueryExpander(build_llm(provider)),
        reranker_factory=lambda provider, threshold: Reranker(
            build_llm(provider), threshold=threshold
        ),
        cache_factory=PgResultCache,
        analytics_factory=SearchAnalytics,
    )
