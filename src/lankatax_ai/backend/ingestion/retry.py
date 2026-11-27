"""Retry helpers for ingestion provider calls."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable

import httpx

from lankatax_ai.backend.ingestion.types import RetryPolicy

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def is_retryable(error: Exception) -> bool:
    """Return whether a provider error is likely transient."""
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int) and status_code in RETRYABLE_STATUS_CODES:
        return True
    return isinstance(error, (httpx.TransportError, TimeoutError))


async def retry_async[T](
    operation: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
) -> T:
    """Run an async operation, retrying transient failures with backoff."""
    attempt = 0
    while True:
        try:
            return await operation()
        except Exception as error:
            if attempt >= policy.max_retries or not is_retryable(error):
                raise
            delay = min(
                policy.backoff_max_seconds,
                policy.backoff_base_seconds * (2**attempt),
            )
            if policy.jitter_ratio:
                delay += random.uniform(0, delay * policy.jitter_ratio)
            logger.warning(
                "Retryable ingestion failure (attempt %d/%d): %s; retrying in %.2fs",
                attempt + 1,
                policy.max_retries + 1,
                error,
                delay,
            )
            await asyncio.sleep(delay)
            attempt += 1
