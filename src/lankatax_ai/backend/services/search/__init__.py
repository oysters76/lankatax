"""Search use case: options, outcome, deps and the ``search`` entry point."""

from __future__ import annotations

from lankatax_ai.backend.services.search.deps import build_search_deps
from lankatax_ai.backend.services.search.service import search
from lankatax_ai.backend.services.search.types import (
    SearchDeps,
    SearchOptions,
    SearchOutcome,
)

__all__ = [
    "SearchDeps",
    "SearchOptions",
    "SearchOutcome",
    "build_search_deps",
    "search",
]
