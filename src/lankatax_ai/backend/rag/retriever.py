from __future__ import annotations

import dataclasses
import logging
from uuid import UUID

from langchain_openai import OpenAIEmbeddings
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from lankatax_ai.backend.db.models import Chunk
from lankatax_ai.backend.rag.metadata_filter import MetadataFilters, apply_metadata_filters
from lankatax_ai.backend.rag.types import RetrievalResult, SearchMode

logger = logging.getLogger(__name__)


class HybridRetriever:
    CANDIDATE_MULTIPLIER: int = 4
    KEYWORD_SEARCH_THRESHOLD = 0.05

    def __init__(
        self, session: AsyncSession, embed_model: OpenAIEmbeddings, *, vector_weight: float = 0.7
    ):
        self.session = session
        self.embed_model = embed_model
        self.vector_weight = vector_weight

    async def search(
        self,
        query: str,
        *,
        mode: SearchMode = SearchMode.hybrid,
        top_k: int = 5,
        filters: MetadataFilters | None = None,
        include_embeddings: bool = False,
    ) -> list[RetrievalResult]:
        filters = filters or MetadataFilters()
        logger.info(
            "Search query=%r mode=%s top_k=%d filters=%s", query, mode.value, top_k, filters
        )

        query_embedding = None
        if mode != SearchMode.keyword:
            query_embedding = await self.embed_model.aembed_query(query)

        if mode == SearchMode.hybrid:
            pool = top_k * self.CANDIDATE_MULTIPLIER
            vector_results = await self._vector_search(
                query_embedding, pool, filters, include_embeddings
            )
            keyword_results = await self._keyword_search(query, pool, filters, include_embeddings)
            results = self._merge_results(vector_results, keyword_results)
        elif mode == SearchMode.vector:
            results = await self._vector_search(query_embedding, top_k, filters, include_embeddings)
        elif mode == SearchMode.keyword:
            results = await self._keyword_search(query, top_k, filters, include_embeddings)
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
        self,
        embedding: list[float],
        top_k: int,
        filters: MetadataFilters,
        include_embeddings: bool = False,
    ) -> list[RetrievalResult]:
        distance = Chunk.embedding.cosine_distance(embedding).label("distance")
        columns = [Chunk.id, Chunk.document_id, Chunk.content, Chunk.metadata_json, distance]
        if include_embeddings:
            columns.append(Chunk.embedding)
        stmt = select(*columns)
        stmt = apply_metadata_filters(stmt, filters)
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
                embedding=list(row["embedding"]) if include_embeddings else None,
            )
            for row in rows
        ]

    async def _keyword_search(
        self, query: str, top_k: int, filters: MetadataFilters, include_embeddings: bool = False
    ) -> list[RetrievalResult]:
        sim = func.similarity(Chunk.content, query).label("sim")
        columns = [Chunk.id, Chunk.document_id, Chunk.content, Chunk.metadata_json, sim]
        if include_embeddings:
            columns.append(Chunk.embedding)
        stmt = select(*columns)
        stmt = apply_metadata_filters(stmt, filters)
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
                embedding=list(row["embedding"]) if include_embeddings else None,
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
            r.score = (self.vector_weight * vs) + ((1 - self.vector_weight) * ks)

        logger.debug(
            "Merge: %d vector + %d keyword → %d unique",
            len(vector_results),
            len(keyword_results),
            len(merged),
        )
        return list(merged.values())
