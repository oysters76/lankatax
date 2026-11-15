from __future__ import annotations

import dataclasses
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


@dataclasses.dataclass
class RetrievalResult:
    chunk_id: UUID
    document_id: UUID
    content: str
    score: float
    vector_score: float | None
    keyword_score: float | None
    metadata: dict
    embedding: list[float] | None = None


class SearchMode(Enum):
    hybrid = "hybrid"
    vector = "vector"
    keyword = "keyword"


class RelevanceScore(BaseModel):
    score: int = Field(description="Relevance score from 1 to 10")


@dataclasses.dataclass
class RerankScorerResult:
    result: RetrievalResult
    score: float
