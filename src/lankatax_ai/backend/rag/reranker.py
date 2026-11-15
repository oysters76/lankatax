from __future__ import annotations

import asyncio
import dataclasses
import logging

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from lankatax_ai.backend.rag.types import RetrievalResult

logger = logging.getLogger(__name__)


class RelevanceScore(BaseModel):
    score: int = Field(description="Relevance score from 1 to 10")


@dataclasses.dataclass
class _ScoredResult:
    result: RetrievalResult
    score: float


RERANK_PROMPT = """Rate how relevant the following text is to answering the query.
Score from 1 to 10, where 1 means completely irrelevant and 10 means directly answers the query.

Query: {query}

Text: {text}"""


class Reranker:
    SEMAPHORE_LIMIT: int = 5

    def __init__(self, llm: BaseChatModel):
        self._llm = llm
        self._structured_llm = self._llm.with_structured_output(RelevanceScore)

    def _build_prompt(self, query: str, result: RetrievalResult) -> str:
        # TODO: replace with versioned YAML prompt from prompt_loader
        return RERANK_PROMPT.format(query=query, text=result.content)

    async def _score_one(
        self, query: str, result: RetrievalResult, sem: asyncio.Semaphore
    ) -> _ScoredResult:
        async with sem:
            prompt = self._build_prompt(query, result)
            response = await self._structured_llm.ainvoke(prompt)
            return _ScoredResult(result=result, score=response.score / 10.0)

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
            reranked = [dataclasses.replace(s.result, score=s.score) for s in scored[:top_k]]
            logger.info(
                "Reranked: top score=%.2f, bottom score=%.2f",
                reranked[0].score if reranked else 0.0,
                reranked[-1].score if reranked else 0.0,
            )
            return reranked

        except Exception:
            logger.warning("Reranker failed, falling back to retriever scores", exc_info=True)
            return results[:top_k]
