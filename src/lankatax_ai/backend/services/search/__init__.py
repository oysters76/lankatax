"""Search use case: options, outcome and the ``search`` entry point."""

from __future__ import annotations

from lankatax_ai.backend.services.search import limits
from lankatax_ai.backend.services.search.service import search
from lankatax_ai.backend.services.search.types import SearchOptions, SearchOutcome

__all__ = ["SearchOptions", "SearchOutcome", "limits", "search"]
