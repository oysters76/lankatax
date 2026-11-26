"""Postgres adapter for the ingestion pipeline."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from lankatax_ai.backend.db.models import Document
from lankatax_ai.backend.documents import (
    get_all_document_ids,
    get_document_by_id,
    get_pending_document_ids,
    update_ingested_hash,
)
from lankatax_ai.backend.ingestion.types import Chunk, IngestionStatus
from lankatax_ai.backend.ingestion.vector_store import (
    delete_chunks_by_document_id,
    store_chunks,
)


class IngestionStore:
    """Session-backed implementation of :class:`IngestionRepository`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_document(self, document_id: UUID) -> Document | None:
        return await get_document_by_id(self._session, document_id)

    async def pending_document_ids(self) -> list[UUID]:
        return await get_pending_document_ids(self._session)

    async def all_document_ids(self) -> list[UUID]:
        return await get_all_document_ids(self._session)

    async def status(self) -> IngestionStatus:
        is_ingested = Document.ingested_hash.is_not(None) & (
            Document.content_hash == Document.ingested_hash
        )
        stmt = select(
            func.count().label("total_documents"),
            func.count(case((is_ingested, 1))).label("ingested"),
            func.count(case((~is_ingested, 1))).label("pending"),
        )
        row = (await self._session.execute(stmt)).one()
        return IngestionStatus(
            total_documents=row.total_documents,
            ingested=row.ingested,
            pending=row.pending,
        )

    async def replace_chunks(self, document_id: UUID, chunks: list[Chunk]) -> int:
        deleted = await delete_chunks_by_document_id(self._session, document_id)
        await store_chunks(self._session, document_id, chunks)
        await self._session.flush()
        return deleted

    async def mark_ingested(self, document_id: UUID, content_hash: str) -> None:
        await update_ingested_hash(self._session, document_id, content_hash)

    async def commit(self) -> None:
        await self._session.commit()

    async def rollback(self) -> None:
        await self._session.rollback()
