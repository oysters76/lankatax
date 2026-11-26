from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

import pytest

from lankatax_ai.backend.cache import CacheStats, PgResultCache
from lankatax_ai.backend.rag.types import RetrievalResult
from lankatax_ai.backend.services.cache import (
    CacheDeps,
    build_cache_deps,
    clear_cache,
    get_cache_stats,
    invalidate_expired,
)

pytestmark = pytest.mark.unit


class _StubSession:
    def __init__(self) -> None:
        self.closed = False
        self.commits = 0

    async def commit(self) -> None:
        self.commits += 1


@asynccontextmanager
async def _session_factory(session: _StubSession):
    try:
        yield session
    finally:
        session.closed = True


class _StubAdmin:
    def __init__(
        self,
        *,
        stats: CacheStats | None = None,
        cleared: int = 0,
        expired: int = 0,
        fail: bool = False,
    ) -> None:
        self._stats = stats or _stats()
        self._cleared = cleared
        self._expired = expired
        self._fail = fail

    async def get_stats(self) -> CacheStats:
        if self._fail:
            raise RuntimeError("cache unavailable")
        return self._stats

    async def clear(self) -> int:
        if self._fail:
            raise RuntimeError("cache unavailable")
        return self._cleared

    async def invalidate_expired(self) -> int:
        if self._fail:
            raise RuntimeError("cache unavailable")
        return self._expired


def _stats(
    *,
    total_entries: int = 4,
    total_hits: int = 6,
    hit_rate: float = 0.6,
    oldest_entry_hours: float = 2.5,
) -> CacheStats:
    return CacheStats(
        total_entries=total_entries,
        total_hits=total_hits,
        hit_rate=hit_rate,
        oldest_entry_hours=oldest_entry_hours,
    )


def _deps(admin: _StubAdmin) -> CacheDeps:
    return CacheDeps(admin_factory=lambda _session: admin)


async def test_stats_are_returned_unchanged():
    expected = _stats()

    result = await get_cache_stats(
        session_factory=lambda: _session_factory(_StubSession()),
        deps=_deps(_StubAdmin(stats=expected)),
    )

    assert result is expected


async def test_stats_do_not_commit_and_close_the_session():
    session = _StubSession()

    await get_cache_stats(
        session_factory=lambda: _session_factory(session),
        deps=_deps(_StubAdmin()),
    )

    assert session.commits == 0
    assert session.closed is True


async def test_clear_returns_count_and_commits():
    session = _StubSession()

    deleted = await clear_cache(
        session_factory=lambda: _session_factory(session),
        deps=_deps(_StubAdmin(cleared=7)),
    )

    assert deleted == 7
    assert session.commits == 1
    assert session.closed is True


async def test_invalidate_returns_count_and_commits():
    session = _StubSession()

    removed = await invalidate_expired(
        session_factory=lambda: _session_factory(session),
        deps=_deps(_StubAdmin(expired=3)),
    )

    assert removed == 3
    assert session.commits == 1
    assert session.closed is True


async def test_admin_receives_the_session():
    session = _StubSession()
    seen: list[object] = []

    def factory(received: object) -> _StubAdmin:
        seen.append(received)
        return _StubAdmin()

    await get_cache_stats(
        session_factory=lambda: _session_factory(session),
        deps=CacheDeps(admin_factory=factory),
    )

    assert seen == [session]


async def test_admin_failure_propagates_and_session_is_closed():
    session = _StubSession()

    with pytest.raises(RuntimeError, match="cache unavailable"):
        await clear_cache(
            session_factory=lambda: _session_factory(session),
            deps=_deps(_StubAdmin(fail=True)),
        )

    assert session.commits == 0
    assert session.closed is True


def test_build_cache_deps_uses_production_store():
    deps = build_cache_deps()

    assert deps.admin_factory is PgResultCache


def test_serialization_round_trip_preserves_results():
    original = RetrievalResult(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content="Payments are due quarterly.",
        score=0.87,
        vector_score=0.9,
        keyword_score=0.4,
        metadata={"page": 3},
        embedding=[0.1, 0.2],
    )

    restored = PgResultCache._deserialize_results(PgResultCache._serialize_results([original]))

    assert len(restored) == 1
    result = restored[0]
    assert result.chunk_id == original.chunk_id
    assert result.document_id == original.document_id
    assert result.content == original.content
    assert result.score == original.score
    assert result.vector_score == original.vector_score
    assert result.keyword_score == original.keyword_score
    assert result.metadata == original.metadata
    assert result.embedding is None


def test_serialization_omits_embeddings():
    result = RetrievalResult(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content="text",
        score=0.5,
        vector_score=None,
        keyword_score=None,
        metadata={},
        embedding=[0.1, 0.2],
    )

    payload = PgResultCache._serialize_results([result])

    assert "embedding" not in payload
