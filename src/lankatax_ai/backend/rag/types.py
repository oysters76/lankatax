from __future__ import annotations

import dataclasses
from enum import Enum
from uuid import UUID


@dataclasses.dataclass
class RetrievalResult:
    chunk_id: UUID
    document_id: UUID
    content: str
    score: float
    vector_score: float | None
    keyword_score: float | None
    metadata: dict


class SearchMode(Enum):
    hybrid = "hybrid"
    vector = "vector"
    keyword = "keyword"
