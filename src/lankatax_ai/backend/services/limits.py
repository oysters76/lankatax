"""Numeric limits and defaults for service request options.

Both transports (and the API's request schemas later) read these instead of
repeating literals. Validation messages are derived from them, so a range only
ever changes in one place.
"""

from __future__ import annotations

MIN_TOP_K = 1
MAX_TOP_K = 50
DEFAULT_TOP_K = 5

MIN_VECTOR_WEIGHT = 0.0
MAX_VECTOR_WEIGHT = 1.0
DEFAULT_VECTOR_WEIGHT = 0.7

MIN_RERANK_THRESHOLD = 1
MAX_RERANK_THRESHOLD = 10
DEFAULT_RERANK_THRESHOLD = 3

MIN_DEDUP_THRESHOLD = 0.0
MAX_DEDUP_THRESHOLD = 1.0
DEFAULT_DEDUP_THRESHOLD = 0.6

MIN_TOKEN_BUDGET = 500
MAX_TOKEN_BUDGET = 16000
