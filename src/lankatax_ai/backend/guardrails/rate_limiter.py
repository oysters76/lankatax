"""Per-key sliding-window rate limiting.

The limiter is in-process. That is enough for a single worker and keeps the
service dependency-free; a shared store can replace the counters when the API
runs behind multiple replicas.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import Callable

from lankatax_ai.backend.guardrails import limits
from lankatax_ai.backend.services.errors import RateLimitError

__all__ = ["RateLimiter"]


class RateLimiter:
    """Allow ``limit`` requests per key in a rolling window."""

    def __init__(
        self,
        limit: int = limits.DEFAULT_RATE_LIMIT,
        window_seconds: float = limits.DEFAULT_RATE_WINDOW_SECONDS,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, key: str) -> None:
        """Record a request or raise :class:`RateLimitError`."""
        now = self._clock()
        async with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= self._window:
                hits.popleft()
            if len(hits) >= self._limit:
                retry_after = max(1, int(self._window - (now - hits[0])))
                raise RateLimitError(
                    f"Too many questions; retry in about {retry_after} seconds",
                    retry_after=retry_after,
                )
            hits.append(now)
