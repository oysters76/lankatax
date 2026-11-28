from __future__ import annotations

from uuid import uuid4

import pytest

from lankatax_ai.backend.rag.context_builder import ContextBuilder
from lankatax_ai.backend.rag.types import RetrievalResult

pytestmark = pytest.mark.unit


def _result(content: str, embedding: list[float]) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content=content,
        score=1.0,
        vector_score=1.0,
        keyword_score=None,
        metadata={
            "document_title": "Guide",
            "from_url": "https://www.ird.gov.lk/guide",
            "document_kind": "pdf",
            "order": 0,
            "page_start": 1,
            "page_end": 1,
        },
        embedding=embedding,
    )


def test_high_threshold_keeps_similar_chunks():
    builder = ContextBuilder(dedup_threshold=0.9)

    context = builder.build([_result("alpha", [1.0, 0.0]), _result("beta", [0.8, 0.6])])

    # Cosine similarity is 0.8: above the old 0.6 default, below the new 0.9.
    assert context.count("[Source:") == 2


def test_low_threshold_collapses_similar_chunks():
    builder = ContextBuilder(dedup_threshold=0.6)

    context = builder.build([_result("alpha", [1.0, 0.0]), _result("beta", [0.8, 0.6])])

    assert context.count("[Source:") == 1
