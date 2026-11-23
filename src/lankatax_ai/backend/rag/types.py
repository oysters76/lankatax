from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any
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
    metadata: dict[str, Any]
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


class SearchSource(Enum):
    cli = "cli"
    api = "api"


class QueryType(Enum):
    factual = "factual"
    calculation = "calculation"
    procedural = "procedural"


@dataclasses.dataclass
class PreprocessedQuery:
    original: str
    expanded: str
    query_type: QueryType


@dataclasses.dataclass
class CacheStats:
    total_entries: int
    total_hits: int
    hit_rate: float
    oldest_entry_hours: float


class ResultCacheBase(ABC):
    @abstractmethod
    async def get(
        self, query_text: str, query_embedding: list[float]
    ) -> list[RetrievalResult] | None: ...

    @abstractmethod
    async def put(
        self, query_text: str, query_embedding: list[float], results: list[RetrievalResult]
    ) -> None: ...

    @abstractmethod
    async def get_stats(self) -> CacheStats: ...

    @abstractmethod
    async def clear(self) -> int: ...

    @abstractmethod
    async def invalidate_expired(self) -> int: ...
