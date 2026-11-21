from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from lankatax_ai.backend.rag.prompts import get_prompt

logger = logging.getLogger(__name__)

DEFAULT_NUM_QUERIES = 3


class ExpandedQueries(BaseModel):
    queries: list[str] = Field(description="Alternative search queries")


class MultiQueryExpander:
    def __init__(
        self,
        llm: BaseChatModel,
        *,
        num_queries: int = DEFAULT_NUM_QUERIES,
        prompt_version: str | None = None,
    ) -> None:
        self._llm = llm
        self._structured_llm = self._llm.with_structured_output(ExpandedQueries)
        self._num_queries = num_queries
        self._prompt = get_prompt("expand", version=prompt_version)

    async def expand(self, query: str) -> list[str]:
        """Generate alternative search queries via LLM. Returns list of rephrasings (excludes original)."""
        try:
            messages = self._prompt.format_messages(num_queries=str(self._num_queries), query=query)
            response = await self._structured_llm.ainvoke(messages)
            alternatives = response.queries[: self._num_queries]
            logger.debug("Expanded query=%r into %d alternatives", query, len(alternatives))
            return alternatives
        except Exception:
            logger.warning("Query expansion failed, using original query only", exc_info=True)
            return []
