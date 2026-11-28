from __future__ import annotations

import pytest

from lankatax_ai.backend.guardrails import QuotaLimiter, RateLimiter, RateLimitRule
from lankatax_ai.backend.services.errors import RateLimitError

pytestmark = pytest.mark.unit


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


async def test_allows_requests_up_to_the_limit():
    clock = _Clock()
    limiter = RateLimiter(limit=3, window_seconds=60, clock=clock)

    for _ in range(3):
        await limiter.check("user")


async def test_blocks_after_the_limit_with_retry_after():
    clock = _Clock()
    limiter = RateLimiter(limit=2, window_seconds=60, clock=clock)
    await limiter.check("user")
    await limiter.check("user")
    clock.now += 10

    with pytest.raises(RateLimitError) as raised:
        await limiter.check("user")

    assert raised.value.retry_after == 50


async def test_window_expiry_releases_the_key():
    clock = _Clock()
    limiter = RateLimiter(limit=1, window_seconds=60, clock=clock)
    await limiter.check("user")
    clock.now += 61

    await limiter.check("user")


async def test_keys_are_isolated():
    clock = _Clock()
    limiter = RateLimiter(limit=1, window_seconds=60, clock=clock)
    await limiter.check("a")

    await limiter.check("b")

    with pytest.raises(RateLimitError):
        await limiter.check("a")


async def test_quota_limiter_enforces_minute_and_daily_rules():
    clock = _Clock()
    limiter = QuotaLimiter(
        [RateLimitRule(2, 60), RateLimitRule(3, 86_400)],
        clock=clock,
    )
    await limiter.check("ip")
    await limiter.check("ip")

    with pytest.raises(RateLimitError) as minute_block:
        await limiter.check("ip")
    assert minute_block.value.retry_after == 60

    clock.now += 61
    await limiter.check("ip")

    with pytest.raises(RateLimitError) as daily_block:
        await limiter.check("ip")
    assert daily_block.value.retry_after > 60


async def test_quota_limiter_applies_global_rules_across_keys():
    clock = _Clock()
    limiter = QuotaLimiter(
        [RateLimitRule(10, 60)],
        global_rules=[RateLimitRule(2, 60)],
        clock=clock,
    )
    await limiter.check("a")
    await limiter.check("b")

    with pytest.raises(RateLimitError):
        await limiter.check("c")


async def test_rejected_requests_do_not_consume_quota():
    clock = _Clock()
    limiter = QuotaLimiter(
        [RateLimitRule(1, 60), RateLimitRule(2, 86_400)],
        clock=clock,
    )
    await limiter.check("ip")

    for _ in range(5):
        with pytest.raises(RateLimitError):
            await limiter.check("ip")

    clock.now += 61
    # Minute freed and the daily bucket still only holds the first request.
    await limiter.check("ip")
