"""Ask use case: options, outcome, deps and the ``ask`` entry point."""

from __future__ import annotations

from lankatax_ai.backend.services.ask.deps import build_ask_deps
from lankatax_ai.backend.services.ask.service import ask
from lankatax_ai.backend.services.ask.types import (
    AnswerGraph,
    AskDeps,
    AskOptions,
    AskOutcome,
    AskRequest,
    AskUsage,
    Citation,
)

__all__ = [
    "AnswerGraph",
    "AskDeps",
    "AskOptions",
    "AskOutcome",
    "AskRequest",
    "AskUsage",
    "Citation",
    "ask",
    "build_ask_deps",
]
