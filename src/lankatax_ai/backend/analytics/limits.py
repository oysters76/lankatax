"""Numeric defaults for the analytics store."""

from __future__ import annotations

# A query counts as a miss when its top score falls below this threshold or it
# returned no results at all.
NO_RESULT_THRESHOLD = 0.3
