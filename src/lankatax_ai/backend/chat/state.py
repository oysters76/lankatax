"""State passed between ask graph nodes."""

from __future__ import annotations

from typing import TypedDict

from lankatax_ai.backend.chat.schemas import AnswerDraft
from lankatax_ai.backend.guardrails.types import (
    GuardrailReason,
    InputGuardResult,
    OutputGuardResult,
)
from lankatax_ai.backend.memory.types import Turn
from lankatax_ai.backend.rag.types import RetrievalResult

__all__ = ["AskState"]


class AskState(TypedDict, total=False):
    """Mutable state of one ask run."""

    question: str
    standalone_query: str
    history: list[Turn]
    input_guard: InputGuardResult
    retrieved: list[RetrievalResult]
    context: str
    draft: AnswerDraft
    output_guard: OutputGuardResult
    answer: str
    refused: bool
    reason: GuardrailReason | None
    attempts: int
    usage: dict[str, int]
    timings: dict[str, int]
