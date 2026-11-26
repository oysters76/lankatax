"""Numeric defaults for the semantic result cache."""

from __future__ import annotations

# Minimum cosine similarity for a cached query to count as a hit.
SIMILARITY_THRESHOLD = 0.95

# Entries older than this are ignored on read and removed by invalidation.
DEFAULT_TTL_HOURS = 24
