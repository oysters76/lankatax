"""Shared guardrail vocabulary: verdicts, reasons and structured LLM results."""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "GuardClassification",
    "GuardrailReason",
    "InputGuardResult",
    "InputVerdict",
    "OutputGuardResult",
    "OutputVerdict",
    "RateLimitRule",
]


class GuardrailReason(Enum):
    """Why a question or answer was refused, rewritten or retried."""

    empty = "empty"
    too_long = "too_long"
    off_topic = "off_topic"
    injection = "injection"
    unsafe = "unsafe"
    pii = "pii"
    insufficient_context = "insufficient_context"
    ungrounded = "ungrounded"
    rate_limited = "rate_limited"


class InputVerdict(Enum):
    allow = "allow"
    refuse = "refuse"
    rewrite = "rewrite"


class OutputVerdict(Enum):
    ok = "pass"
    retry = "retry"
    refuse = "refuse"


@dataclasses.dataclass(frozen=True, slots=True)
class InputGuardResult:
    """Outcome of checking one question."""

    verdict: InputVerdict
    reason: GuardrailReason | None = None
    question: str = ""
    detail: str = ""

    @property
    def allowed(self) -> bool:
        return self.verdict is InputVerdict.allow


@dataclasses.dataclass(frozen=True, slots=True)
class OutputGuardResult:
    """Outcome of checking one drafted answer."""

    verdict: OutputVerdict
    reason: GuardrailReason | None = None
    answer: str = ""
    unsupported: tuple[str, ...] = ()
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.verdict is OutputVerdict.ok


class GuardClassification(BaseModel):
    """Structured verdict from the input guard model."""

    allowed: bool = Field(description="Whether the question is safe and in scope")
    category: Literal["in_scope", "off_topic", "injection", "unsafe", "pii"] = "in_scope"
    reason: str = Field(default="", description="Short internal rationale")
    rewritten_question: str | None = Field(
        default=None,
        description="A neutral rewrite when the question is allowed but needs normalising",
    )


class GroundingCheck(BaseModel):
    """Structured verdict from the output guard model."""

    grounded: bool = Field(description="Whether every claim is supported by the context")
    unsupported_claims: list[str] = Field(default_factory=list)
    reason: str = Field(default="", description="Short internal rationale")


@dataclasses.dataclass(frozen=True, slots=True)
class RateLimitRule:
    """One quota: at most ``limit`` requests per ``window_seconds``."""

    limit: int
    window_seconds: float

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("limit must be at least 1")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
