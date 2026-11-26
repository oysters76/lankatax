"""Compatibility re-exports for the frozen CLI.

The store now lives in :mod:`lankatax_ai.backend.analytics`. This module keeps
the old import path working until the CLI is rewritten in Phase 6.
"""

from __future__ import annotations

from lankatax_ai.backend.analytics import (
    SearchAnalytics,
    SearchStats,
    SearchTiming,
    StatsSnapshot,
)

__all__ = ["SearchAnalytics", "SearchStats", "SearchTiming", "StatsSnapshot"]
