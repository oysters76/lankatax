"""Request, outcome and collaborator types for the ask use case.

Transport-neutral: the future HTTP adapter translates into and out of these
types, and the LangGraph implementation arrives through ``AskDeps``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from lankatax_ai.backend.config import settings
from lankatax_ai.backend.guardrails.rate_limiter import RateLimiter
from lankatax_ai.backend.guardrails.types import GuardrailReason
from lankatax_ai.backend.memory.types import ConversationStore
from lankatax_ai.backend.rag.types import RetrievalResult
from lankatax_ai.backend.services.ask import limits
from lankatax_ai.backend.services.errors import ValidationError
from lankatax_ai.backend.services.utils import require_range
from lankatax_ai.backend.types import LlmProvider, SearchMode

__all__ = [
    "AnswerGraph",
    "AskDeps",
    "AskOptions",
    "AskOutcome",
    "AskRequest",
    "AskUsage",
    "Citation",
]


@dataclasses.dataclass(frozen=True, slots=True)
class AskOptions:
    """Everything an ask run needs besides the question and conversation."""

    provider: LlmProvider = dataclasses.field(default_factory=lambda: settings.ask_llm_provider)
    top_k: int = dataclasses.field(default_factory=lambda: settings.ask_top_k)
    mode: SearchMode = SearchMode.hybrid
    rerank: bool = True
    memory_turns: int = dataclasses.field(default_factory=lambda: settings.ask_memory_turns)
    guard_input: bool = dataclasses.field(default_factory=lambda: settings.ask_guard_input)
    guard_output: bool = dataclasses.field(default_factory=lambda: settings.ask_guard_output)
    min_score: float = dataclasses.field(default_factory=lambda: settings.ask_min_score)
    answer_max_tokens: int = dataclasses.field(
        default_factory=lambda: settings.ask_answer_max_tokens
    )
    context_token_budget: int = dataclasses.field(
        default_factory=lambda: settings.ask_context_token_budget
    )

    def __post_init__(self) -> None:
        require_range("Top-k", self.top_k, limits.MIN_TOP_K, limits.MAX_TOP_K)
        require_range("Memory turns", self.memory_turns, 0, limits.MAX_MEMORY_TURNS)
        if not 0.0 <= self.min_score <= 1.0:
            raise ValidationError("min_score must be between 0 and 1")
        if self.answer_max_tokens < 1:
            raise ValidationError("answer_max_tokens must be positive")
        if self.context_token_budget < 1:
            raise ValidationError("context_token_budget must be positive")


@dataclasses.dataclass(frozen=True, slots=True)
class AskRequest:
    """One question, optionally continuing an existing conversation."""

    question: str
    conversation_id: str | None = None
    fingerprint: str | None = None
    requested_by: str | None = None
    options: AskOptions = dataclasses.field(default_factory=AskOptions)


@dataclasses.dataclass(frozen=True, slots=True)
class Citation:
    """One retrieved chunk referenced by the answer."""

    chunk_id: str
    document_id: str
    title: str
    url: str | None
    snippet: str
    score: float


@dataclasses.dataclass(frozen=True, slots=True)
class AskUsage:
    """Cheap, provider-independent usage counters."""

    llm_calls: int = 0
    answer_chars: int = 0
    retrieved_chunks: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class AskOutcome:
    """Result of one ask run, independent of how it will be displayed."""

    conversation_id: str
    answer: str
    citations: tuple[Citation, ...]
    refused: bool
    reason: GuardrailReason | None
    usage: AskUsage
    timings: dict[str, int]
    retrieved: tuple[RetrievalResult, ...] = ()
    trace_url: str | None = None


class AnswerGraph(Protocol):
    """The orchestration port; the LangGraph adapter implements it."""

    async def ainvoke(self, state: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclasses.dataclass(frozen=True, slots=True)
class AskDeps:
    """Collaborators wired by the caller; the service builds nothing."""

    graph_factory: Callable[[AskOptions], AnswerGraph]
    conversation_store: ConversationStore
    rate_limiter: RateLimiter
