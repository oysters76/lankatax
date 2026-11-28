"""Per-key sliding-window rate limiting, optionally with global quotas.

The limiter is in-process. That is enough for a single worker and keeps the
service dependency-free; a shared store can replace the counters when the API
runs behind multiple replicas.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import Callable, Sequence

from lankatax_ai.backend.guardrails import limits
from lankatax_ai.backend.guardrails.types import RateLimitRule
from lankatax_ai.backend.services.errors import RateLimitError

_CLEANUP_EVERY = 1024

__all__ = ["QuotaLimiter", "RateLimitRule", "RateLimiter"]


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


class QuotaLimiter:
    """Enforce several windows per caller plus several windows globally.

    A request must satisfy every rule. Rejected requests do not consume quota,
    so a caller blocked by a daily cap cannot keep extending a short window.
    """

    def __init__(
        self,
        rules: Sequence[RateLimitRule],
        *,
        global_rules: Sequence[RateLimitRule] = (),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._rules = tuple(rules)
        self._global_rules = tuple(global_rules)
        self._clock = clock
        # Only successful requests create buckets, so rejected keys cannot grow
        # memory. Stale buckets are dropped periodically.
        self._hits: dict[tuple[str, int], deque[float]] = {}
        self._global_hits: dict[int, deque[float]] = {}
        self._checks = 0
        self._lock = asyncio.Lock()

    async def check(self, key: str) -> None:
        """Record a request for ``key`` or raise :class:`RateLimitError`."""
        now = self._clock()
        async with self._lock:
            self._checks += 1
            if self._checks % _CLEANUP_EVERY == 0:
                self._cleanup(now)

            waits = self._waits(
                [self._hits.get((key, index), deque()) for index in range(len(self._rules))],
                self._rules,
                now,
            )
            waits.extend(
                self._waits(
                    [
                        self._global_hits.get(index, deque())
                        for index in range(len(self._global_rules))
                    ],
                    self._global_rules,
                    now,
                )
            )
            if waits:
                retry_after = max(waits)
                raise RateLimitError(
                    f"Too many questions; retry in about {retry_after} seconds",
                    retry_after=retry_after,
                )
            for index, _rule in enumerate(self._rules):
                self._hits.setdefault((key, index), deque()).append(now)
            for index, _rule in enumerate(self._global_rules):
                self._global_hits.setdefault(index, deque()).append(now)

    def _cleanup(self, now: float) -> None:
        max_window = max(
            (rule.window_seconds for rule in (*self._rules, *self._global_rules)),
            default=0.0,
        )
        for bucket_key, bucket in list(self._hits.items()):
            if not bucket or now - bucket[-1] >= max_window:
                del self._hits[bucket_key]
        for index, bucket in list(self._global_hits.items()):
            if not bucket or now - bucket[-1] >= max_window:
                del self._global_hits[index]

    @staticmethod
    def _waits(
        buckets: Sequence[deque[float]],
        rules: Sequence[RateLimitRule],
        now: float,
    ) -> list[int]:
        waits: list[int] = []
        for bucket, rule in zip(buckets, rules, strict=True):
            while bucket and now - bucket[0] >= rule.window_seconds:
                bucket.popleft()
            if len(bucket) >= rule.limit:
                waits.append(max(1, int(rule.window_seconds - (now - bucket[0]))))
        return waits
