from __future__ import annotations

import logging

import tiktoken

from lankatax_ai.backend.rag.types import RetrievalResult
from lankatax_ai.backend.utils import cosine_similarity

logger = logging.getLogger(__name__)


class ContextBuilder:
    def __init__(
        self,
        token_budget: int = 3000,
        dedup_threshold: float = 0.6,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.token_budget = token_budget
        self.dedup_threshold = dedup_threshold
        self._embedding_model = embedding_model
        self._encoder = tiktoken.encoding_for_model(embedding_model)

    def build(self, results: list[RetrievalResult]) -> str:
        deduped = self._deduplicate(results)
        ordered = self._order_chunks(deduped)

        context_parts: list[str] = []
        tokens_used = 0

        for r in ordered:
            formatted = self._format_chunk(r)
            chunk_tokens = self._count_tokens(formatted)
            if tokens_used + chunk_tokens > self.token_budget:
                logger.debug(
                    "Token budget reached (%d/%d), stopping", tokens_used, self.token_budget
                )
                break
            context_parts.append(formatted)
            tokens_used += chunk_tokens

        logger.info("Built context: %d chunks, %d tokens", len(context_parts), tokens_used)
        return "\n\n".join(context_parts)

    def _deduplicate(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        kept: list[RetrievalResult] = []
        for r in results:
            is_duplicate = False
            for existing in kept:
                if r.embedding is not None and existing.embedding is not None:
                    sim = cosine_similarity(r.embedding, existing.embedding)
                    if sim > self.dedup_threshold:
                        is_duplicate = True
                        break
            if not is_duplicate:
                kept.append(r)
        return kept

    def _order_chunks(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        return sorted(results, key=lambda r: (r.document_id, r.metadata.get("order", 0)))

    def _count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def _format_chunk(self, result: RetrievalResult) -> str:
        title = result.metadata.get("document_title", "Unknown")
        url = result.metadata.get("from_url", "")
        kind = result.metadata.get("document_kind", "page")
        page_start = result.metadata.get("page_start")
        page_end = result.metadata.get("page_end")

        parts = [title]
        if kind == "pdf" and page_start is not None:
            if page_end and page_end != page_start:
                parts.append(f"p.{page_start}-{page_end}")
            else:
                parts.append(f"p.{page_start}")
        if url:
            parts.append(url)

        source_line = f"[Source: {', '.join(parts)}]"
        return f"{source_line}\n{result.content}"
