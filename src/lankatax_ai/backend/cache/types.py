"""Value types and ports for the semantic result cache.

The search use case reads and writes entries; the cache use case reports on
them and prunes them. The two ports below keep each caller's surface minimal and
let both depend on the store through protocols rather than the SQL adapter.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol

from lankatax_ai.backend.rag.types import RetrievalResult


@dataclasses.dataclass
class CacheStats:
    total_entries: int
    total_hits: int
    hit_rate: float
    oldest_entry_hours: float


class ResultCache(Protocol):
    async def get(
        self, query_text: str, query_embedding: list[float]
    ) -> list[RetrievalResult] | None: ...

    async def put(
        self, query_text: str, query_embedding: list[float], results: list[RetrievalResult]
    ) -> None: ...


class CacheAdmin(Protocol):
    async def get_stats(self) -> CacheStats: ...

    async def clear(self) -> int: ...

    async def invalidate_expired(self) -> int: ...
