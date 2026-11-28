from __future__ import annotations

from types import SimpleNamespace

import pytest

from lankatax_ai.backend.jobs.contracts import CacheExpireInput
from lankatax_ai.backend.jobs.tasks import cache as cache_task

pytestmark = pytest.mark.unit


class _Ctx:
    def __init__(self) -> None:
        self.logs: list[object] = []

    def log(self, line: object) -> None:
        self.logs.append(line)


async def test_run_cache_expire_reports_count(monkeypatch):
    async def fake_invalidate(**kwargs):
        return 7

    monkeypatch.setattr(cache_task, "invalidate_expired", fake_invalidate)
    ctx = _Ctx()
    runtime = SimpleNamespace(session_factory=None, cache=None)

    output = await cache_task.run_cache_expire(
        CacheExpireInput(),
        ctx,
        runtime,  # type: ignore[arg-type]
    )

    assert output.expired == 7
    assert output.duration_ms >= 0
    assert ctx.logs
