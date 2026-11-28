"""Types for the chat orchestration: state, runner and graph dependencies."""

from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable
from typing import TypedDict

from langchain_core.language_models import BaseChatModel

from lankatax_ai.backend.chat.schemas import AnswerDraft
from lankatax_ai.backend.guardrails.input_validator import InputValidator
from lankatax_ai.backend.guardrails.output_checker import OutputChecker
from lankatax_ai.backend.guardrails.types import (
    GuardrailReason,
    InputGuardResult,
    OutputGuardResult,
)
from lankatax_ai.backend.memory.types import Turn
from lankatax_ai.backend.rag.types import RetrievalResult
from lankatax_ai.backend.services.search.types import SearchOutcome

__all__ = ["AskState", "GraphConfig", "GraphDeps", "SearchRunner"]


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


SearchRunner = Callable[[str], Awaitable[SearchOutcome]]


@dataclasses.dataclass(frozen=True, slots=True)
class GraphConfig:
    """Guardrail and prompt choices for one graph build."""

    min_score: float = 0.2
    guard_input: bool = True
    guard_output: bool = True
    answer_prompt_version: str | None = None
    condense_prompt_version: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class GraphDeps:
    """Everything the graph needs; built by the ask use case wiring."""

    llm: BaseChatModel
    search: SearchRunner
    config: GraphConfig = dataclasses.field(default_factory=GraphConfig)
    input_validator: InputValidator | None = None
    output_checker: OutputChecker | None = None
