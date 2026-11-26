from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from lankatax_ai.backend.analytics import SearchAnalytics, StatsSnapshot
from lankatax_ai.backend.services.stats import (
    SearchStats,
    StatsDeps,
    build_stats_deps,
    get_stats,
)

pytestmark = pytest.mark.unit


class _StubSession:
    def __init__(self) -> None:
        self.closed = False


@asynccontextmanager
async def _session_factory(session: _StubSession):
    try:
        yield session
    finally:
        session.closed = True


class _StubReader:
    def __init__(self, snapshot: StatsSnapshot, *, fail: bool = False) -> None:
        self._snapshot = snapshot
        self._fail = fail
        self.calls: list[dict[str, float]] = []

    async def snapshot(self, *, threshold: float) -> StatsSnapshot:
        self.calls.append({"threshold": threshold})
        if self._fail:
            raise RuntimeError("analytics unavailable")
        return self._snapshot


def _snapshot(
    *,
    total_queries: int = 10,
    avg_latency_ms: float = 120.0,
    avg_top_score: float = 0.75,
    cache_hit_count: int = 3,
    multi_query_count: int = 2,
    hit_count: int = 8,
    no_result_queries: list[str] | None = None,
) -> StatsSnapshot:
    return StatsSnapshot(
        total_queries=total_queries,
        avg_latency_ms=avg_latency_ms,
        avg_top_score=avg_top_score,
        cache_hit_count=cache_hit_count,
        multi_query_count=multi_query_count,
        hit_count=hit_count,
        no_result_queries=["unmatched one", "unmatched two"]
        if no_result_queries is None
        else no_result_queries,
    )


def _deps(reader: _StubReader, *, threshold: float = 0.3) -> StatsDeps:
    return StatsDeps(reader_factory=lambda _session: reader, no_result_threshold=threshold)


async def _run(reader: _StubReader, *, threshold: float = 0.3) -> tuple[SearchStats, _StubSession]:
    session = _StubSession()
    stats = await get_stats(
        session_factory=lambda: _session_factory(session),
        deps=_deps(reader, threshold=threshold),
    )
    return stats, session


async def test_derives_hit_rate_from_snapshot():
    reader = _StubReader(_snapshot())

    stats, _ = await _run(reader)

    assert stats.total_queries == 10
    assert stats.hit_rate == pytest.approx(0.8)


async def test_copies_every_metric():
    reader = _StubReader(_snapshot())

    stats, _ = await _run(reader)

    assert stats.avg_latency_ms == 120.0
    assert stats.avg_top_score == 0.75
    assert stats.cache_hit_count == 3
    assert stats.multi_query_count == 2
    assert stats.no_result_queries == ["unmatched one", "unmatched two"]


async def test_empty_dataset_has_zero_hit_rate():
    reader = _StubReader(
        _snapshot(
            total_queries=0,
            hit_count=0,
            avg_latency_ms=0.0,
            avg_top_score=0.0,
            cache_hit_count=0,
            multi_query_count=0,
            no_result_queries=[],
        )
    )

    stats, _ = await _run(reader)

    assert stats.total_queries == 0
    assert stats.hit_rate == 0.0


async def test_hit_rate_is_zero_when_nothing_hit():
    reader = _StubReader(_snapshot(hit_count=0))

    stats, _ = await _run(reader)

    assert stats.hit_rate == 0.0


async def test_forwards_configured_threshold():
    reader = _StubReader(_snapshot())

    await _run(reader, threshold=0.42)

    assert reader.calls == [{"threshold": 0.42}]


async def test_reader_receives_the_session_and_session_is_closed():
    reader = _StubReader(_snapshot())
    session = _StubSession()
    seen: list[object] = []

    def factory(received: object) -> _StubReader:
        seen.append(received)
        return reader

    await get_stats(
        session_factory=lambda: _session_factory(session),
        deps=StatsDeps(reader_factory=factory, no_result_threshold=0.3),
    )

    assert seen == [session]
    assert session.closed is True


async def test_no_result_queries_are_copied():
    source = ["one"]
    reader = _StubReader(_snapshot(no_result_queries=source))

    stats, _ = await _run(reader)

    source.append("two")
    assert stats.no_result_queries == ["one"]


async def test_reader_failure_propagates():
    reader = _StubReader(_snapshot(), fail=True)

    with pytest.raises(RuntimeError, match="analytics unavailable"):
        await _run(reader)


def test_build_stats_deps_uses_production_reader_and_threshold():
    deps = build_stats_deps()

    assert deps.reader_factory is SearchAnalytics
    assert deps.no_result_threshold > 0
