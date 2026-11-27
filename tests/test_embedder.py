from __future__ import annotations

import pytest

from lankatax_ai.backend.ingestion.embedder import embed_chunks
from lankatax_ai.backend.ingestion.types import EmbeddingPolicy, RetryPolicy
from tests.helpers import make_chunk

pytestmark = pytest.mark.unit


class _RateLimitError(Exception):
    status_code = 429


class _BadRequestError(Exception):
    status_code = 400


class _FakeEmbeddings:
    def __init__(
        self, failures: list[Exception] | None = None, vectors_per_call: int | None = None
    ):
        self.calls: list[list[str]] = []
        self._failures = list(failures or [])
        self._vectors_per_call = vectors_per_call

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self._failures:
            raise self._failures.pop(0)
        count = self._vectors_per_call if self._vectors_per_call is not None else len(texts)
        return [[float(index)] * 3 for index in range(count)]


def _policy(
    *,
    max_batch_tokens: int = 8000,
    max_retries: int = 3,
) -> EmbeddingPolicy:
    return EmbeddingPolicy(
        max_batch_tokens=max_batch_tokens,
        retry=RetryPolicy(
            max_retries=max_retries,
            backoff_base_seconds=0,
            backoff_max_seconds=0,
            jitter_ratio=0,
        ),
    )


async def test_batches_chunks_by_token_budget():
    chunks = [
        make_chunk(content="a", token_count=4, embedding=None),
        make_chunk(content="b", token_count=4, embedding=None),
        make_chunk(content="c", token_count=4, embedding=None),
    ]
    model = _FakeEmbeddings()

    result = await embed_chunks(model, chunks, policy=_policy(max_batch_tokens=8))

    assert model.calls == [["a", "b"], ["c"]]
    assert len(result) == 3
    assert all(chunk.embedding is not None for chunk in result)


async def test_oversized_chunk_is_embedded_alone():
    chunks = [make_chunk(content="large", token_count=100, embedding=None)]
    model = _FakeEmbeddings()

    result = await embed_chunks(model, chunks, policy=_policy(max_batch_tokens=8))

    assert model.calls == [["large"]]
    assert result[0].embedding is not None


async def test_retries_transient_failures():
    chunks = [make_chunk(content="a", token_count=1, embedding=None)]
    model = _FakeEmbeddings(failures=[_RateLimitError(), _RateLimitError()])

    result = await embed_chunks(model, chunks, policy=_policy(max_retries=2))

    assert len(model.calls) == 3
    assert result[0].embedding is not None


async def test_does_not_retry_bad_requests():
    chunks = [make_chunk(content="a", token_count=1, embedding=None)]
    model = _FakeEmbeddings(failures=[_BadRequestError()])

    with pytest.raises(_BadRequestError):
        await embed_chunks(model, chunks, policy=_policy(max_retries=3))

    assert len(model.calls) == 1


async def test_raises_after_retries_are_exhausted():
    chunks = [make_chunk(content="a", token_count=1, embedding=None)]
    model = _FakeEmbeddings(failures=[_RateLimitError(), _RateLimitError(), _RateLimitError()])

    with pytest.raises(_RateLimitError):
        await embed_chunks(model, chunks, policy=_policy(max_retries=2))

    assert len(model.calls) == 3


async def test_rejects_mismatched_vector_count():
    chunks = [
        make_chunk(content="a", token_count=1, embedding=None),
        make_chunk(content="b", token_count=1, embedding=None),
    ]
    model = _FakeEmbeddings(vectors_per_call=1)

    with pytest.raises(RuntimeError, match="returned 1 vectors for 2 chunks"):
        await embed_chunks(model, chunks, policy=_policy())


async def test_empty_chunks_do_not_call_the_provider():
    model = _FakeEmbeddings()

    assert await embed_chunks(model, [], policy=_policy()) == []
    assert model.calls == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_retries": -1},
        {"backoff_base_seconds": -1},
        {"backoff_max_seconds": -1},
        {"jitter_ratio": 1.1},
        {"jitter_ratio": -0.1},
    ],
)
def test_retry_policy_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)


def test_embedding_policy_rejects_invalid_batch_size():
    with pytest.raises(ValueError):
        EmbeddingPolicy(max_batch_tokens=0)
