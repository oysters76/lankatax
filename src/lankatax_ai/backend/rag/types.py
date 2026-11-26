from __future__ import annotations

import dataclasses
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from lankatax_ai.backend.types import QueryType, SearchMode, SearchSource

__all__ = [
    "PreprocessedQuery",
    "QueryType",
    "RelevanceScore",
    "RerankScorerResult",
    "RetrievalResult",
    "SearchMode",
    "SearchSource",
]


@dataclasses.dataclass
class RetrievalResult:
    chunk_id: UUID
    document_id: UUID
    content: str
    score: float
    vector_score: float | None
    keyword_score: float | None
    metadata: dict[str, Any]
    embedding: list[float] | None = None


class RelevanceScore(BaseModel):
    score: int = Field(description="Relevance score from 1 to 10")


@dataclasses.dataclass
class RerankScorerResult:
    result: RetrievalResult
    score: float


@dataclasses.dataclass
class PreprocessedQuery:
    original: str
    expanded: str
    query_type: QueryType
