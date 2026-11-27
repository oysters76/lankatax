from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import pytest

from lankatax_ai.backend.ingestion.store import IngestionStore
from lankatax_ai.backend.ingestion.types import Chunk, EmbeddingPolicy, IngestionStatus
from lankatax_ai.backend.services.errors import ValidationError
from lankatax_ai.backend.services.ingest import (
    IngestDeps,
    IngestOptions,
    IngestTarget,
    build_ingest_deps,
    get_ingestion_status,
    ingest,
)
from tests.helpers import make_document

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


class _StubEmbeddings:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.1] * 4 for _ in texts]


class _StubRepository:
    def __init__(
        self,
        documents: list | None = None,
        *,
        status: IngestionStatus | None = None,
        fail_ids: set[UUID] | None = None,
    ) -> None:
        self._documents = {doc.id: doc for doc in documents or []}
        self._status = status or IngestionStatus(total_documents=0, pending=0, ingested=0)
        self._fail_ids = fail_ids or set()
        self.commits = 0
        self.rollbacks = 0
        self.replaced: list[tuple[UUID, list[Chunk]]] = []
        self.marked: list[tuple[UUID, str]] = []

    async def get_document(self, document_id: UUID):
        if document_id in self._fail_ids:
            raise RuntimeError("database unavailable")
        return self._documents.get(document_id)

    async def pending_document_ids(self) -> list[UUID]:
        return list(self._documents)

    async def all_document_ids(self) -> list[UUID]:
        return list(self._documents)

    async def status(self) -> IngestionStatus:
        return self._status

    async def replace_chunks(self, document_id: UUID, chunks: list[Chunk]) -> int:
        self.replaced.append((document_id, chunks))
        return 0

    async def mark_ingested(self, document_id: UUID, content_hash: str) -> None:
        self.marked.append((document_id, content_hash))

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1


def _doc(**overrides):
    defaults = {
        "id": uuid4(),
        "raw_text": "PAYE tax is deducted by employer. Employees must file returns.",
        "raw_html": "<p>PAYE tax is deducted by employer.</p><p>Employees must file returns.</p>",
    }
    return make_document(**(defaults | overrides))


def _deps(repo: _StubRepository, embeddings: _StubEmbeddings | None = None) -> IngestDeps:
    return IngestDeps(
        repository_factory=lambda _session: repo,
        embeddings=embeddings or _StubEmbeddings(),
        embedding_policy=EmbeddingPolicy(),
    )


async def test_ingests_pending_documents():
    docs = [_doc(url="https://www.ird.gov.lk/a"), _doc(url="https://www.ird.gov.lk/b")]
    repo = _StubRepository(docs)
    embeddings = _StubEmbeddings()

    result = await ingest(
        IngestOptions(target=IngestTarget.pending),
        session_factory=lambda: _session_factory(_StubSession()),
        deps=_deps(repo, embeddings),
    )

    assert result.docs_processed == 2
    assert result.chunks_created > 0
    assert result.tokens_used > 0
    assert result.errors == []
    assert repo.commits == 2
    assert repo.rollbacks == 0
    assert len(repo.replaced) == 2
    assert len(repo.marked) == 2
    assert embeddings.calls


async def test_ingests_all_documents():
    docs = [_doc(url="https://www.ird.gov.lk/a")]
    repo = _StubRepository(docs)

    result = await ingest(
        IngestOptions(target=IngestTarget.all),
        session_factory=lambda: _session_factory(_StubSession()),
        deps=_deps(repo),
    )

    assert result.docs_processed == 1
    assert repo.commits == 1


async def test_ingests_single_document():
    doc = _doc()
    repo = _StubRepository([doc])

    result = await ingest(
        IngestOptions(target=IngestTarget.document, document_id=doc.id),
        session_factory=lambda: _session_factory(_StubSession()),
        deps=_deps(repo),
    )

    assert result.docs_processed == 1
    assert repo.commits == 1
    assert repo.marked == [(doc.id, doc.content_hash)]


async def test_missing_document_returns_error_without_commit():
    repo = _StubRepository([])
    missing = uuid4()

    result = await ingest(
        IngestOptions(target=IngestTarget.document, document_id=missing),
        session_factory=lambda: _session_factory(_StubSession()),
        deps=_deps(repo),
    )

    assert result.docs_processed == 0
    assert result.errors == [f"Document {missing} not found"]
    assert repo.commits == 0


async def test_failed_document_rolls_back_and_keeps_going():
    ok = _doc(url="https://www.ird.gov.lk/ok")
    broken = _doc(url="https://www.ird.gov.lk/broken")
    repo = _StubRepository([ok, broken], fail_ids={broken.id})

    result = await ingest(
        IngestOptions(target=IngestTarget.all),
        session_factory=lambda: _session_factory(_StubSession()),
        deps=_deps(repo),
    )

    assert result.docs_processed == 1
    assert len(result.errors) == 1
    assert "database unavailable" in result.errors[0]
    assert repo.commits == 1
    assert repo.rollbacks == 1


async def test_progress_callback_reports_each_document():
    docs = [_doc(url="https://www.ird.gov.lk/a"), _doc(url="https://www.ird.gov.lk/b")]
    repo = _StubRepository(docs)
    events: list[tuple] = []

    await ingest(
        IngestOptions(target=IngestTarget.pending),
        session_factory=lambda: _session_factory(_StubSession()),
        deps=_deps(repo),
        on_progress=lambda *event: events.append(event),
    )

    assert len(events) == 2
    assert all(event[4] is None for event in events)


async def test_session_is_closed():
    session = _StubSession()

    await ingest(
        IngestOptions(target=IngestTarget.pending),
        session_factory=lambda: _session_factory(session),
        deps=_deps(_StubRepository([])),
    )

    assert session.closed is True


async def test_status_is_returned_without_commit():
    expected = IngestionStatus(total_documents=5, pending=2, ingested=3)
    repo = _StubRepository(status=expected)

    status = await get_ingestion_status(
        session_factory=lambda: _session_factory(_StubSession()),
        deps=_deps(repo),
    )

    assert status is expected
    assert repo.commits == 0


def test_document_target_requires_document_id():
    with pytest.raises(ValidationError, match="document_id is required"):
        IngestOptions(target=IngestTarget.document)


def test_batch_target_rejects_document_id():
    with pytest.raises(ValidationError, match="only valid when target is 'document'"):
        IngestOptions(target=IngestTarget.pending, document_id=uuid4())


def test_build_ingest_deps_uses_production_store():
    embeddings = _StubEmbeddings()

    deps = build_ingest_deps(embeddings)

    assert deps.repository_factory is IngestionStore
    assert deps.embeddings is embeddings
