from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import pytest

from lankatax_ai.backend.rag.metadata_filter import MetadataFilters
from lankatax_ai.backend.rag.types import RetrievalResult
from lankatax_ai.backend.services.errors import ValidationError
from lankatax_ai.backend.services.search import SearchDeps, SearchOptions, search
from lankatax_ai.backend.types import LlmProvider, SearchMode, SearchSource

pytestmark = pytest.mark.unit

QUERY = "APIT rates"


class _StubSession:
    async def commit(self) -> None:
        pass


@asynccontextmanager
async def _session_factory():
    yield _StubSession()


def _result(
    *,
    content: str = "APIT is calculated on monthly remuneration.",
    score: float = 0.5,
    chunk_id: UUID | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id or uuid4(),
        document_id=uuid4(),
        content=content,
        score=score,
        vector_score=score,
        keyword_score=None,
        metadata={
            "document_title": "APIT Tax Tables",
            "from_url": "https://www.ird.gov.lk/apit.pdf",
        },
    )


def _options(**overrides) -> SearchOptions:
    return SearchOptions(source=SearchSource.api, **overrides)


class _StubEmbeddings:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def aembed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        return [0.1, 0.2, 0.3]


class _StubRetriever:
    def __init__(self, result_sets: list[list[RetrievalResult]]) -> None:
        self._result_sets = result_sets
        self.calls: list[dict[str, object]] = []

    async def search(
        self,
        query: str,
        *,
        mode: SearchMode,
        top_k: int,
        filters: MetadataFilters | None,
        include_embeddings: bool,
    ) -> list[RetrievalResult]:
        self.calls.append(
            {
                "query": query,
                "mode": mode,
                "top_k": top_k,
                "filters": filters,
                "include_embeddings": include_embeddings,
            }
        )
        index = min(len(self.calls) - 1, len(self._result_sets) - 1)
        return list(self._result_sets[index])


class _StubExpander:
    def __init__(self, queries: list[str]) -> None:
        self._queries = queries
        self.calls: list[str] = []

    async def expand(self, query: str) -> list[str]:
        self.calls.append(query)
        return list(self._queries)


class _StubReranker:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results
        self.calls: list[tuple[str, int]] = []

    async def rerank(
        self, query: str, results: list[RetrievalResult], *, top_k: int
    ) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        return list(self._results)


class _StubCache:
    def __init__(self, get_result: list[RetrievalResult] | None = None) -> None:
        self._get_result = get_result
        self.get_calls: list[tuple[str, list[float]]] = []
        self.put_calls: list[tuple[str, list[float], list[RetrievalResult]]] = []

    async def get(
        self, query_text: str, query_embedding: list[float]
    ) -> list[RetrievalResult] | None:
        self.get_calls.append((query_text, query_embedding))
        return self._get_result

    async def put(
        self, query_text: str, query_embedding: list[float], results: list[RetrievalResult]
    ) -> None:
        self.put_calls.append((query_text, query_embedding, results))


class _StubAnalytics:
    def __init__(self, fail: bool = False) -> None:
        self._fail = fail
        self.calls: list[dict[str, object]] = []

    async def log_search(self, **kwargs: object) -> None:
        if self._fail:
            raise RuntimeError("analytics unavailable")
        self.calls.append(kwargs)


async def _run(
    retriever: _StubRetriever,
    *,
    query: str = QUERY,
    options: SearchOptions | None = None,
    expander: _StubExpander | None = None,
    reranker: _StubReranker | None = None,
    cache: _StubCache | None = None,
    analytics: _StubAnalytics | None = None,
    embeddings: _StubEmbeddings | None = None,
):
    active_cache = cache if cache is not None else _StubCache()
    active_analytics = analytics if analytics is not None else _StubAnalytics()
    return await search(
        query,
        options or _options(),
        session_factory=_session_factory,
        deps=SearchDeps(
            embeddings=embeddings or _StubEmbeddings(),
            retriever_factory=lambda _session, _options: retriever,
            expander_factory=lambda _provider: expander,
            reranker_factory=lambda _provider, _threshold: reranker,
            cache_factory=lambda _session: active_cache,
            analytics_factory=lambda _session: active_analytics,
        ),
    )


async def test_returns_retriever_results():
    results = [_result(), _result()]
    retriever = _StubRetriever([results])

    outcome = await _run(retriever)

    assert outcome.results == results
    assert outcome.cache_hit is False
    assert outcome.context is None
    assert outcome.processed_query.original == QUERY


async def test_forwards_options_to_retriever():
    retriever = _StubRetriever([[_result()]])
    filters = MetadataFilters(category="tax_info", doc_type="pdf")

    await _run(
        retriever,
        options=_options(mode=SearchMode.vector, top_k=3, filters=filters),
    )

    call = retriever.calls[0]
    assert call["mode"] is SearchMode.vector
    assert call["top_k"] == 3
    assert call["filters"] is filters
    assert call["include_embeddings"] is False


async def test_include_context_builds_context_and_requests_embeddings():
    retriever = _StubRetriever([[_result()]])

    outcome = await _run(retriever, options=_options(include_context=True))

    assert outcome.context is not None
    assert "APIT is calculated on monthly remuneration." in outcome.context
    assert retriever.calls[0]["include_embeddings"] is True


async def test_empty_query_is_rejected():
    retriever = _StubRetriever([[]])

    with pytest.raises(ValidationError, match="Query must not be empty"):
        await _run(retriever, query="   ")

    assert retriever.calls == []


async def test_multi_query_merges_and_dedupes():
    shared = uuid4()
    low = _result(score=0.2, chunk_id=shared)
    high = _result(score=0.9, chunk_id=shared)
    other = _result(score=0.7)
    retriever = _StubRetriever([[low], [high], [other]])
    expander = _StubExpander(["APIT tax table", "monthly withholding"])

    outcome = await _run(retriever, options=_options(multi_query=True), expander=expander)

    assert [call["query"] for call in retriever.calls] == [
        outcome.processed_query.expanded,
        "APIT tax table",
        "monthly withholding",
    ]
    assert [r.chunk_id for r in outcome.results] == [high.chunk_id, other.chunk_id]
    assert outcome.multi_query is True


async def test_multi_query_is_off_by_default():
    retriever = _StubRetriever([[_result()]])
    expander = _StubExpander(["unused"])

    await _run(retriever, expander=expander)

    assert expander.calls == []
    assert len(retriever.calls) == 1


async def test_rerank_uses_original_query():
    results = [_result(), _result()]
    reranked = [_result(content="reranked", score=0.95)]
    retriever = _StubRetriever([results])
    reranker = _StubReranker(reranked)

    outcome = await _run(retriever, options=_options(rerank=True), reranker=reranker)

    assert outcome.results == reranked
    assert reranker.calls == [(QUERY, 5)]


async def test_no_results_skips_rerank_and_context():
    retriever = _StubRetriever([[]])
    reranker = _StubReranker([_result()])

    outcome = await _run(
        retriever,
        options=_options(rerank=True, include_context=True),
        reranker=reranker,
    )

    assert outcome.has_results is False
    assert outcome.context is None
    assert reranker.calls == []


async def test_timings_are_recorded():
    retriever = _StubRetriever([[_result()]])

    outcome = await _run(retriever)

    assert outcome.timings.total_ms >= 0
    assert outcome.timings.elapsed_seconds >= 0.0


async def test_cache_hit_skips_retrieval_and_rerank():
    cached = [_result(score=0.99)]
    retriever = _StubRetriever([[_result()]])
    reranker = _StubReranker([_result()])
    cache = _StubCache(get_result=cached)
    analytics = _StubAnalytics()

    outcome = await _run(
        retriever,
        options=_options(rerank=True, include_context=True),
        reranker=reranker,
        cache=cache,
        analytics=analytics,
    )

    assert outcome.cache_hit is True
    assert outcome.results == cached
    assert outcome.context is not None
    assert retriever.calls == []
    assert reranker.calls == []
    assert cache.put_calls == []
    assert analytics.calls[0]["cache_hit"] is True
    assert analytics.calls[0]["top_score"] == 0.99


async def test_cache_miss_writes_results():
    results = [_result()]
    retriever = _StubRetriever([results])
    cache = _StubCache()

    outcome = await _run(retriever, cache=cache)

    assert outcome.cache_hit is False
    assert len(cache.get_calls) == 1
    assert len(cache.put_calls) == 1
    query_text, embedding, stored = cache.put_calls[0]
    assert query_text == outcome.processed_query.expanded
    assert embedding == [0.1, 0.2, 0.3]
    assert stored == results


async def test_cache_disabled_skips_lookup_and_write():
    retriever = _StubRetriever([[_result()]])
    cache = _StubCache()
    embeddings = _StubEmbeddings()

    await _run(
        retriever,
        options=_options(use_cache=False),
        cache=cache,
        embeddings=embeddings,
    )

    assert cache.get_calls == []
    assert cache.put_calls == []
    assert embeddings.calls == []


async def test_cache_write_is_skipped_when_empty():
    retriever = _StubRetriever([[]])
    cache = _StubCache()
    analytics = _StubAnalytics()

    outcome = await _run(retriever, cache=cache, analytics=analytics)

    assert outcome.has_results is False
    assert cache.put_calls == []
    assert analytics.calls[0]["top_score"] is None
    assert analytics.calls[0]["result_count"] == 0


async def test_analytics_records_source_and_provider_only_when_reranked():
    analytics = _StubAnalytics()
    reranker = _StubReranker([_result()])

    await _run(
        _StubRetriever([[_result()]]),
        options=_options(rerank=True),
        reranker=reranker,
        analytics=analytics,
    )

    assert analytics.calls[0]["source"] is SearchSource.api
    assert analytics.calls[0]["llm_provider"] is LlmProvider.gemini

    without_rerank = _StubAnalytics()
    await _run(_StubRetriever([[_result()]]), analytics=without_rerank)
    assert without_rerank.calls[0]["llm_provider"] is None


async def test_analytics_failure_propagates():
    retriever = _StubRetriever([[_result()]])

    with pytest.raises(RuntimeError, match="analytics unavailable"):
        await _run(retriever, analytics=_StubAnalytics(fail=True))
