"""Types for the search-stats use case.

The result types and the reader port live with the analytics store, because the
store and both of its callers share them. This module only adds the wiring the
stats use case needs.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from sqlalchemy.ext.asyncio import AsyncSession

from lankatax_ai.backend.analytics.types import SearchStats, StatsReader, StatsSnapshot

__all__ = [
    "SearchStats",
    "StatsDeps",
    "StatsReader",
    "StatsReaderFactory",
    "StatsSnapshot",
]


StatsReaderFactory = Callable[[AsyncSession], StatsReader]


@dataclasses.dataclass(frozen=True, slots=True)
class StatsDeps:
    """Collaborators the stats use case needs, wired by the caller."""

    reader_factory: StatsReaderFactory
    no_result_threshold: float
