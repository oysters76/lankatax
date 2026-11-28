"""Structured outputs used by the ask graph."""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["AnswerDraft", "CondensedQuestion"]


class AnswerDraft(BaseModel):
    """The model's answer plus the context blocks it relied on."""

    answer: str = Field(description="Grounded answer with [n] citations, or a refusal")
    citation_numbers: list[int] = Field(
        default_factory=list,
        description="1-based numbers of the retrieved context blocks used",
    )
    insufficient_context: bool = Field(
        default=False,
        description="True when the context does not contain the answer",
    )


class CondensedQuestion(BaseModel):
    """A follow-up question rewritten as a standalone search query."""

    question: str = Field(description="Self-contained search query")
