from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_NUM_QUERIES = 3

EXPAND_PROMPT = """You are a query expansion assistant for a Sri Lanka tax knowledge base.
The knowledge base contains documents from the Inland Revenue Department (ird.gov.lk):
tax guides, circulars, gazette notifications, APIT tables, VAT/SSCL/WHT rules, and procedural guides.

Given the user's search query, generate {num_queries} alternative search queries that would
help find relevant documents. Each alternative should:
- Approach the topic from a different angle (rephrase, use synonyms, or focus on a different aspect)
- Use specific Sri Lankan tax terminology where appropriate
- Target different document types (e.g. one for a guide, one for a table, one for a circular)

Do NOT answer the query. Only generate search queries.

Original query: {query}"""


class ExpandedQueries(BaseModel):
    queries: list[str] = Field(description="Alternative search queries")


class MultiQueryExpander:
    def __init__(self, llm: BaseChatModel, *, num_queries: int = DEFAULT_NUM_QUERIES) -> None:
        self._llm = llm
        self._structured_llm = self._llm.with_structured_output(ExpandedQueries)
        self._num_queries = num_queries

    def _build_prompt(self, query: str) -> str:
        # TODO: replace with versioned YAML prompt from prompt_loader
        return EXPAND_PROMPT.format(num_queries=self._num_queries, query=query)

    async def expand(self, query: str) -> list[str]:
        """Generate alternative search queries via LLM. Returns list of rephrasings (excludes original)."""
        try:
            prompt = self._build_prompt(query)
            response = await self._structured_llm.ainvoke(prompt)
            alternatives = response.queries[: self._num_queries]
            logger.debug("Expanded query=%r into %d alternatives", query, len(alternatives))
            return alternatives
        except Exception:
            logger.warning("Query expansion failed, using original query only", exc_info=True)
            return []
