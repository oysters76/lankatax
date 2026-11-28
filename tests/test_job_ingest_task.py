from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest

from lankatax_ai.backend.ingestion.types import EmbeddingPolicy, IngestBatch
from lankatax_ai.backend.jobs.contracts import IngestDocumentInput, IngestInput
from lankatax_ai.backend.jobs.tasks import ingest as ingest_task

pytestmark = pytest.mark.unit


class _Ctx:
    def __init__(self, cancelled: bool = False) -> None:
        self.is_cancelled = cancelled
        self.cancelled = False
        self.refreshes = 0
        self.logs: list[object] = []
        self.workflow_run_id = "run-1"

    def refresh_timeout(self, _duration) -> None:
        self.refreshes += 1

    def log(self, line: object) -> None:
        self.logs.append(line)

    async def aio_cancel(self) -> None:
        self.cancelled = True


class _Repo:
    def __init__(self, ids: list) -> None:
        self.ids = ids

    async def pending_document_ids(self) -> list:
        return list(self.ids)

    async def all_document_ids(self) -> list:
        return list(self.ids)


def _runtime(ids: list) -> SimpleNamespace:
    @asynccontextmanager
    async def session_factory():
        yield object()

    return SimpleNamespace(
        session_factory=session_factory,
        ingest=SimpleNamespace(
            repository_factory=lambda _session: _Repo(ids),
            embeddings=object(),
            embedding_policy=EmbeddingPolicy(),
        ),
    )


def _batch(
    *,
    document_id=None,
    chunks: int = 1,
    tokens: int = 10,
    error: str | None = None,
    budget: bool = False,
) -> IngestBatch:
    return IngestBatch(
        document_id=document_id or uuid4(),
        title="doc",
        chunks_created=chunks,
        tokens_used=tokens,
        error=error,
        budget_exhausted=budget,
    )


async def test_run_ingest_aggregates_and_finalizes(monkeypatch):
    ids = [uuid4(), uuid4()]

    async def fake_batches(*args, **kwargs):
        yield _batch(document_id=ids[0], chunks=2, tokens=20)
        yield _batch(document_id=ids[1], chunks=1, tokens=10)

    finalized: list[str | None] = []

    async def fake_finalize(*, session_factory, run_id=None, ingested_at=None):
        finalized.append(run_id)
        return 1

    monkeypatch.setattr(ingest_task, "ingest_batches", fake_batches)
    monkeypatch.setattr(ingest_task, "finalize_ingest", fake_finalize)

    output = await ingest_task.run_ingest(
        IngestInput(),
        _Ctx(),
        _runtime(ids),  # type: ignore[arg-type]
    )

    assert output.docs_processed == 2
    assert output.chunks_created == 3
    assert output.tokens_used == 30
    assert output.partial is False
    assert finalized == ["run-1"]


async def test_run_ingest_reports_budget_and_errors(monkeypatch):
    ids = [uuid4()]

    async def fake_batches(*args, **kwargs):
        yield _batch(
            document_id=ids[0],
            error="token budget exhausted",
            budget=True,
        )

    async def fake_finalize(**kwargs):
        return 1

    monkeypatch.setattr(ingest_task, "ingest_batches", fake_batches)
    monkeypatch.setattr(ingest_task, "finalize_ingest", fake_finalize)

    output = await ingest_task.run_ingest(
        IngestInput(),
        _Ctx(),
        _runtime(ids),  # type: ignore[arg-type]
    )

    assert output.docs_processed == 0
    assert output.budget_exhausted is True
    assert output.partial is True
    assert output.errors == [f"{ids[0]}: token budget exhausted"]


async def test_run_ingest_finalizes_after_cancellation(monkeypatch):
    ids = [uuid4(), uuid4()]

    async def fake_batches(*args, **kwargs):
        yield _batch(document_id=ids[0])
        yield _batch(document_id=ids[1])

    finalized: list[str | None] = []

    async def fake_finalize(*, run_id=None, **kwargs):
        finalized.append(run_id)
        return 1

    monkeypatch.setattr(ingest_task, "ingest_batches", fake_batches)
    monkeypatch.setattr(ingest_task, "finalize_ingest", fake_finalize)
    ctx = _Ctx(cancelled=True)

    output = await ingest_task.run_ingest(
        IngestInput(),
        ctx,
        _runtime(ids),  # type: ignore[arg-type]
    )

    assert output.docs_processed == 0
    assert ctx.cancelled is True
    assert finalized == ["run-1"]


async def test_run_ingest_document_uses_the_given_id(monkeypatch):
    document_id = uuid4()
    seen: list[list] = []

    async def fake_batches(repo, doc_ids, *args, **kwargs):
        seen.append(list(doc_ids))
        yield _batch(document_id=document_id)

    async def fake_finalize(**kwargs):
        return 1

    monkeypatch.setattr(ingest_task, "ingest_batches", fake_batches)
    monkeypatch.setattr(ingest_task, "finalize_ingest", fake_finalize)

    output = await ingest_task.run_ingest_document(
        IngestDocumentInput(document_id=document_id),
        _Ctx(),
        _runtime([]),  # type: ignore[arg-type]
    )

    assert output.docs_processed == 1
    assert seen == [[document_id]]
