"""Compatibility re-exports for the frozen CLI.

The cache now lives in :mod:`lankatax_ai.backend.cache`. This module keeps the
old import path working until the CLI is rewritten in Phase 6.
"""

from __future__ import annotations

from lankatax_ai.backend.cache import (
    DEFAULT_TTL_HOURS,
    SIMILARITY_THRESHOLD,
    PgResultCache,
)

__all__ = ["DEFAULT_TTL_HOURS", "SIMILARITY_THRESHOLD", "PgResultCache"]
