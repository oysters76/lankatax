from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest
from langchain_openai import OpenAIEmbeddings
from sqlalchemy import select

from lankatax_ai.backend.db.models import Document
from lankatax_ai.backend.ingestion.vector_store import store_chunks
from lankatax_ai.backend.rag.retriever import HybridRetriever
from lankatax_ai.backend.scraper.storage import save_documents
from lankatax_ai.backend.types import SearchMode
from tests.helpers import make_chunk, make_document

pytestmark = pytest.mark.db

QUERY = "APIT tax"
FILLER = "APIT tax table row {i}: monthly remuneration and withholding rates."


class _StubEmbeddings:
    """Stand-in for OpenAIEmbeddings so tests never hit the network."""

    def __init__(self, vector: list[float] | None = None) -> None:
        self._vector = vector
        self.calls = 0

    async def aembed_query(self, _: str) -> list[float]:
        self.calls += 1
        if self._vector is None:
            raise AssertionError("keyword mode must not call the embedding model")
        return self._vector


def _retriever(session, stub: _StubEmbeddings) -> HybridRetriever:
    return HybridRetriever(session, cast(OpenAIEmbeddings, stub))


async def _seed(session, *, count: int, with_embeddings: bool = False) -> None:
    doc = make_document(url="https://www.ird.gov.lk/pool-test", title="APIT Tax Tables")
    await save_documents(session, [doc])
    await session.commit()
    doc_id = (
        await session.execute(select(Document.id).where(Document.url == doc.url))
    ).scalar_one()

    base = make_chunk()
    chunks = [
        make_chunk(
            content=FILLER.format(i=i),
            citation=replace(base.citation, order=i),
            embedding=[0.1] * 1536 if with_embeddings else None,
        )
        for i in range(count)
    ]
    await store_chunks(session, doc_id, chunks)
    await session.commit()


async def test_default_returns_top_k(clean_session):
    await _seed(clean_session, count=30)
    stub = _StubEmbeddings()

    results = await _retriever(clean_session, stub).search(QUERY, mode=SearchMode.keyword, top_k=5)

    assert len(results) == 5
    assert stub.calls == 0


async def test_candidate_k_returns_full_pool(clean_session):
    await _seed(clean_session, count=30)

    results = await _retriever(clean_session, _StubEmbeddings()).search(
        QUERY, mode=SearchMode.keyword, top_k=5, candidate_k=20
    )

    assert len(results) == 20


async def test_candidate_k_applies_to_vector_mode(clean_session):
    await _seed(clean_session, count=30, with_embeddings=True)
    stub = _StubEmbeddings([0.1] * 1536)

    results = await _retriever(clean_session, stub).search(
        QUERY, mode=SearchMode.vector, top_k=5, candidate_k=20
    )

    assert len(results) == 20
    assert stub.calls == 1


async def test_candidate_k_smaller_than_top_k_is_rejected(clean_session):
    await _seed(clean_session, count=30)

    with pytest.raises(ValueError, match="candidate_k"):
        await _retriever(clean_session, _StubEmbeddings()).search(
            QUERY, mode=SearchMode.keyword, top_k=5, candidate_k=3
        )
