"""Input guardrails: cheap heuristics first, then an in-scope LLM classifier.

The validator returns a verdict instead of raising, so the orchestration layer
decides how to refuse. Only deterministic checks run when no model is wired, and
a classifier failure falls back to the heuristic verdict (fail open) so a
provider outage does not block every question.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel

from lankatax_ai.backend.guardrails import limits
from lankatax_ai.backend.guardrails.constants import (
    CATEGORY_REASONS,
    INJECTION_PATTERNS,
    PII_PATTERNS,
)
from lankatax_ai.backend.guardrails.types import (
    GuardClassification,
    GuardrailReason,
    InputGuardResult,
    InputVerdict,
)
from lankatax_ai.backend.rag.prompts import get_prompt

__all__ = ["InputValidator", "redact_pii"]

logger = logging.getLogger(__name__)


def redact_pii(text: str) -> str:
    """Mask common Sri Lankan identifiers before text is logged, stored or sent."""
    for pattern, replacement in PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class InputValidator:
    """Validate a question before retrieval."""

    def __init__(
        self,
        *,
        llm: BaseChatModel | None = None,
        max_chars: int = limits.DEFAULT_MAX_QUESTION_CHARS,
        prompt_version: str | None = None,
    ) -> None:
        self._max_chars = max_chars
        self._prompt = get_prompt("guard_input", version=prompt_version) if llm else None
        self._structured = (
            llm.with_structured_output(GuardClassification) if llm is not None else None
        )

    async def validate(self, question: str, *, history: str = "") -> InputGuardResult:
        cleaned = redact_pii(question.strip())
        if not cleaned:
            return InputGuardResult(InputVerdict.refuse, GuardrailReason.empty)
        if len(cleaned) > self._max_chars:
            return InputGuardResult(
                InputVerdict.refuse,
                GuardrailReason.too_long,
                question=cleaned,
                detail=f"over {self._max_chars} characters",
            )
        for pattern in INJECTION_PATTERNS:
            if pattern.search(cleaned):
                return InputGuardResult(
                    InputVerdict.refuse,
                    GuardrailReason.injection,
                    question=cleaned,
                    detail="prompt-injection pattern",
                )

        if self._structured is None or self._prompt is None:
            return InputGuardResult(InputVerdict.allow, question=cleaned)

        try:
            messages = self._prompt.format_messages(question=cleaned, history=history)
            raw = await self._structured.ainvoke(messages)
            classification = GuardClassification.model_validate(raw)
        except Exception:
            logger.warning("Input guard classifier failed; allowing the question", exc_info=True)
            return InputGuardResult(InputVerdict.allow, question=cleaned)

        if not classification.allowed:
            return InputGuardResult(
                InputVerdict.refuse,
                CATEGORY_REASONS.get(classification.category, GuardrailReason.off_topic),
                question=cleaned,
                detail=classification.reason,
            )

        rewritten = (classification.rewritten_question or "").strip()
        if rewritten and rewritten != cleaned:
            return InputGuardResult(InputVerdict.rewrite, question=rewritten)

        return InputGuardResult(InputVerdict.allow, question=cleaned)
