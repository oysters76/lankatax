from __future__ import annotations

import dataclasses
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from lankatax_ai.backend.db.models import CacheEntry
from lankatax_ai.backend.rag.types import CacheStats, ResultCacheBase, RetrievalResult

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.95
DEFAULT_TTL_HOURS = 24


class PgResultCache(ResultCacheBase):
    def __init__(
        self,
        session: AsyncSession,
        *,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        ttl_hours: int = DEFAULT_TTL_HOURS,
    ) -> None:
        self._session = session
        self._similarity_threshold = similarity_threshold
        self._ttl_hours = ttl_hours

    async def get(
        self, query_text: str, query_embedding: list[float]
    ) -> list[RetrievalResult] | None:
        """Look up cached results by semantic similarity. Returns None on miss."""
        distance = CacheEntry.query_embedding.cosine_distance(query_embedding).label("distance")
        max_distance = 1.0 - self._similarity_threshold
        cutoff = datetime.now(UTC) - timedelta(hours=self._ttl_hours)

        stmt = (
            select(CacheEntry.id, CacheEntry.response, distance)
            .where(CacheEntry.created_at >= cutoff)
            .order_by(distance)
            .limit(1)
        )
        row = (await self._session.execute(stmt)).mappings().first()

        if row is None:
            logger.debug("Cache miss query=%r (empty cache)", query_text)
            return None
        if row["distance"] > max_distance:
            logger.debug(
                "Cache miss query=%r (nearest distance=%.4f, threshold=%.4f)",
                query_text,
                row["distance"],
                max_distance,
            )
            return None

        await self._session.execute(
            update(CacheEntry)
            .where(CacheEntry.id == row["id"])
            .values(hit_count=CacheEntry.hit_count + 1)
        )

        logger.debug("Cache hit query=%r (distance=%.4f)", query_text, row["distance"])
        return self._deserialize_results(row["response"])

    async def put(
        self, query_text: str, query_embedding: list[float], results: list[RetrievalResult]
    ) -> None:
        """Cache search results for a query."""
        entry = CacheEntry(
            query_text=query_text,
            query_embedding=query_embedding,
            response=self._serialize_results(results),
        )
        self._session.add(entry)
        logger.debug("Cached %d results for query=%r", len(results), query_text)

    async def get_stats(self) -> CacheStats:
        """Compute cache statistics."""
        summary = (
            (
                await self._session.execute(
                    select(
                        func.count().label("total"),
                        func.coalesce(func.sum(CacheEntry.hit_count), 0).label("hits"),
                        func.min(CacheEntry.created_at).label("oldest"),
                    )
                )
            )
            .mappings()
            .one()
        )

        total = summary["total"]
        hits = summary["hits"]

        if total == 0:
            return CacheStats(total_entries=0, total_hits=0, hit_rate=0.0, oldest_entry_hours=0.0)

        age = datetime.now(UTC) - summary["oldest"].replace(tzinfo=UTC)

        return CacheStats(
            total_entries=total,
            total_hits=hits,
            hit_rate=hits / (hits + total) if (hits + total) > 0 else 0.0,
            oldest_entry_hours=age.total_seconds() / 3600,
        )

    async def clear(self) -> int:
        """Delete all cache entries. Returns count deleted."""
        result = await self._session.execute(delete(CacheEntry))
        deleted = cast(CursorResult[Any], result).rowcount
        logger.debug("Cleared %d cache entries", deleted)
        return deleted

    async def invalidate_expired(self) -> int:
        """Remove entries older than TTL. Returns count removed."""
        cutoff = datetime.now(UTC) - timedelta(hours=self._ttl_hours)
        result = await self._session.execute(
            delete(CacheEntry).where(CacheEntry.created_at < cutoff)
        )
        deleted = cast(CursorResult[Any], result).rowcount
        logger.debug("Invalidated %d expired cache entries", deleted)
        return deleted

    @staticmethod
    def _serialize_results(results: list[RetrievalResult]) -> str:
        """Convert results to JSON string for storage."""
        rows = []
        for r in results:
            d = dataclasses.asdict(r)
            d.pop("embedding", None)
            d["chunk_id"] = str(d["chunk_id"])
            d["document_id"] = str(d["document_id"])
            rows.append(d)
        return json.dumps(rows)

    @staticmethod
    def _deserialize_results(data: str) -> list[RetrievalResult]:
        """Reconstruct results from cached JSON string."""
        rows = json.loads(data)
        return [
            RetrievalResult(
                chunk_id=UUID(r["chunk_id"]),
                document_id=UUID(r["document_id"]),
                content=r["content"],
                score=r["score"],
                vector_score=r.get("vector_score"),
                keyword_score=r.get("keyword_score"),
                metadata=r.get("metadata", {}),
            )
            for r in rows
        ]
