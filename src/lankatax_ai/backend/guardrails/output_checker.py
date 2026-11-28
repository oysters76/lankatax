"""Output guardrails: citation integrity, grounding and safety checks.

The checker never decides the final refusal text; it reports ``retry`` or
``refuse`` so the orchestration layer can make one corrective attempt before
falling back to a safe refusal.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel

from lankatax_ai.backend.guardrails import limits
from lankatax_ai.backend.guardrails.constants import CITATION_RE, DANGEROUS_HTML_RE
from lankatax_ai.backend.guardrails.input_validator import redact_pii
from lankatax_ai.backend.guardrails.types import (
    GroundingCheck,
    GuardrailReason,
    OutputGuardResult,
    OutputVerdict,
)
from lankatax_ai.backend.rag.prompts import get_prompt

__all__ = ["OutputChecker"]

logger = logging.getLogger(__name__)


class OutputChecker:
    """Check one drafted answer before it reaches the caller."""

    def __init__(
        self,
        *,
        llm: BaseChatModel | None = None,
        prompt_version: str | None = None,
    ) -> None:
        self._prompt = get_prompt("guard_output", version=prompt_version) if llm else None
        self._structured = llm.with_structured_output(GroundingCheck) if llm is not None else None

    async def check(
        self,
        *,
        question: str,
        answer: str,
        context: str,
        citation_count: int,
    ) -> OutputGuardResult:
        sanitized = redact_pii(answer.strip())
        if not sanitized:
            return OutputGuardResult(OutputVerdict.refuse, GuardrailReason.empty)

        if len(sanitized) > limits.MAX_ANSWER_CHARS:
            return OutputGuardResult(
                OutputVerdict.refuse,
                GuardrailReason.too_long,
                detail=f"answer over {limits.MAX_ANSWER_CHARS} characters",
            )

        if DANGEROUS_HTML_RE.search(sanitized):
            return OutputGuardResult(
                OutputVerdict.refuse,
                GuardrailReason.unsafe,
                detail="answer contains raw HTML",
            )

        invalid = [
            number
            for number in (int(match) for match in CITATION_RE.findall(sanitized))
            if number < 1 or number > citation_count
        ]
        if invalid:
            return OutputGuardResult(
                OutputVerdict.retry,
                GuardrailReason.ungrounded,
                answer=sanitized,
                detail=f"unknown citation numbers: {sorted(set(invalid))}",
            )

        if self._structured is None or self._prompt is None:
            return OutputGuardResult(OutputVerdict.ok, answer=sanitized)

        try:
            messages = self._prompt.format_messages(
                question=question,
                answer=sanitized,
                context=context,
            )
            raw = await self._structured.ainvoke(messages)
            grounding = GroundingCheck.model_validate(raw)
        except Exception:
            logger.warning("Output guard classifier failed; accepting the answer", exc_info=True)
            return OutputGuardResult(OutputVerdict.ok, answer=sanitized)

        if not grounding.grounded:
            return OutputGuardResult(
                OutputVerdict.retry,
                GuardrailReason.ungrounded,
                answer=sanitized,
                unsupported=tuple(grounding.unsupported_claims),
                detail=grounding.reason,
            )

        return OutputGuardResult(OutputVerdict.ok, answer=sanitized)
