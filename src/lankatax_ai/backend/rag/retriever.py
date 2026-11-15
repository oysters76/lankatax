from __future__ import annotations

import dataclasses
import logging
from uuid import UUID

from langchain_openai import OpenAIEmbeddings
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from lankatax_ai.backend.db.models import Chunk, Document
from lankatax_ai.backend.rag.types import RetrievalResult, SearchMode

logger = logging.getLogger(__name__)


class HybridRetriever:
    CANDIDATE_MULTIPLIER: int = 4
    KEYWORD_SEARCH_THRESHOLD = 0.05
    VECTOR_WEIGHT = 0.7

    def __init__(self, session: AsyncSession, embed_model: OpenAIEmbeddings):
        self.session = session
        self.embed_model = embed_model

    async def search(
        self,
        query: str,
        *,
        mode: SearchMode = SearchMode.hybrid,
        top_k: int = 5,
        category: str | None = None,
    ) -> list[RetrievalResult]:
        logger.info(
            "Search query=%r mode=%s top_k=%d category=%s", query, mode.value, top_k, category
        )

        query_embedding = None
        if mode != SearchMode.keyword:
            query_embedding = await self.embed_model.aembed_query(query)

        if mode == SearchMode.hybrid:
            pool = top_k * self.CANDIDATE_MULTIPLIER
            vector_results = await self._vector_search(query_embedding, pool, category)
            keyword_results = await self._keyword_search(query, pool, category)
            results = self._merge_results(vector_results, keyword_results)
        elif mode == SearchMode.vector:
            results = await self._vector_search(query_embedding, top_k, category)
        elif mode == SearchMode.keyword:
            results = await self._keyword_search(query, top_k, category)
        else:
            raise ValueError(f"Unknown search mode: {mode}")

        results = sorted(results, key=lambda r: r.score, reverse=True)[:top_k]
        logger.info(
            "Returning %d results (top score=%.3f)",
            len(results),
            results[0].score if results else 0.0,
        )
        return results

    async def _vector_search(
        self, embedding: list[float], top_k: int, category: str | None
    ) -> list[RetrievalResult]:
        distance = Chunk.embedding.cosine_distance(embedding).label("distance")
        stmt = select(
            Chunk.id,
            Chunk.document_id,
            Chunk.content,
            Chunk.metadata_json,
            distance,
        )
        if category is not None:
            stmt = stmt.join(Document, Chunk.document_id == Document.id)
            stmt = stmt.where(Document.category == category)

        stmt = stmt.order_by(distance).limit(top_k)
        rows = (await self.session.execute(stmt)).mappings().all()
        logger.debug("Vector search returned %d candidates", len(rows))
        return [
            RetrievalResult(
                chunk_id=row["id"],
                document_id=row["document_id"],
                content=row["content"],
                score=max(0.0, 1.0 - row["distance"]),
                vector_score=max(0.0, 1.0 - row["distance"]),
                keyword_score=None,
                metadata=row["metadata_json"] or {},
            )
            for row in rows
        ]

    async def _keyword_search(
        self, query: str, top_k: int, category: str | None
    ) -> list[RetrievalResult]:
        sim = func.similarity(Chunk.content, query).label("sim")
        stmt = select(
            Chunk.id,
            Chunk.document_id,
            Chunk.content,
            Chunk.metadata_json,
            sim,
        )

        if category is not None:
            stmt = stmt.join(Document, Chunk.document_id == Document.id)
            stmt = stmt.where(Document.category == category)

        stmt = stmt.where(func.similarity(Chunk.content, query) > self.KEYWORD_SEARCH_THRESHOLD)
        stmt = stmt.order_by(sim.desc()).limit(top_k)

        rows = (await self.session.execute(stmt)).mappings().all()
        logger.debug("Keyword search returned %d candidates", len(rows))
        return [
            RetrievalResult(
                chunk_id=row["id"],
                document_id=row["document_id"],
                content=row["content"],
                score=float(row["sim"]),
                vector_score=None,
                keyword_score=float(row["sim"]),
                metadata=row["metadata_json"] or {},
            )
            for row in rows
        ]

    def _merge_results(
        self, vector_results: list[RetrievalResult], keyword_results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        merged: dict[UUID, RetrievalResult] = {}
        for r in vector_results:
            merged[r.chunk_id] = dataclasses.replace(r)
        for r in keyword_results:
            if r.chunk_id in merged:
                merged[r.chunk_id].keyword_score = r.keyword_score
            else:
                merged[r.chunk_id] = dataclasses.replace(r)
        for r in merged.values():
            vs = r.vector_score or 0.0
            ks = r.keyword_score or 0.0
            r.score = (self.VECTOR_WEIGHT * vs) + ((1 - self.VECTOR_WEIGHT) * ks)

        logger.debug(
            "Merge: %d vector + %d keyword → %d unique",
            len(vector_results),
            len(keyword_results),
            len(merged),
        )
        return list(merged.values())
