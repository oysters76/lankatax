from __future__ import annotations

import asyncio
import dataclasses
import logging

from langchain_core.language_models import BaseChatModel

from lankatax_ai.backend.rag.prompts import get_prompt
from lankatax_ai.backend.rag.types import RelevanceScore, RerankScorerResult, RetrievalResult

logger = logging.getLogger(__name__)


class Reranker:
    SEMAPHORE_LIMIT: int = 5

    def __init__(
        self, llm: BaseChatModel, *, threshold: int = 3, prompt_version: str | None = None
    ):
        self._llm = llm
        self._structured_llm = self._llm.with_structured_output(RelevanceScore)
        self._threshold = threshold
        self._prompt = get_prompt("rerank", version=prompt_version)

    async def _score_one(
        self, query: str, result: RetrievalResult, sem: asyncio.Semaphore
    ) -> RerankScorerResult:
        async with sem:
            messages = self._prompt.format_messages(query=query, text=result.content)
            response = await self._structured_llm.ainvoke(messages)
            return RerankScorerResult(result=result, score=response.score / 10.0)

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        *,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        logger.info("Reranking %d results for query=%r", len(results), query)
        try:
            sem = asyncio.Semaphore(self.SEMAPHORE_LIMIT)
            scored = list(await asyncio.gather(*[self._score_one(query, r, sem) for r in results]))
            scored.sort(key=lambda s: s.score, reverse=True)
            threshold_norm = self._threshold / 10.0
            reranked = [
                dataclasses.replace(s.result, score=s.score)
                for s in scored[:top_k]
                if s.score >= threshold_norm
            ]
            logger.info(
                "Reranked: top score=%.2f, bottom score=%.2f",
                reranked[0].score if reranked else 0.0,
                reranked[-1].score if reranked else 0.0,
            )
            return reranked

        except Exception:
            logger.warning("Reranker failed, falling back to retriever scores", exc_info=True)
            return results[:top_k]
