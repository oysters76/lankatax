"""Persistence for the shared ``documents`` table.

Scraping and ingestion both operate on documents, but with different write
shapes. Keeping the table access here makes that shared boundary explicit
instead of hiding it inside the scraper package.

``DocumentStore`` is the injectable facade; the module-level functions are kept
as thin wrappers for the frozen CLI and for existing callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import case, func, literal_column, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from lankatax_ai.backend.db.models import Document


@dataclass
class ScrapeStatusRecord:
    total: int
    recent: int
    first_scrape: datetime
    last_scrape: datetime


class DocumentStore:
    """Session-backed access to the shared ``documents`` table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, document_id: UUID) -> Document | None:
        return await self._session.get(Document, document_id)

    async def save(self, documents: list[Document]) -> None:
        for doc in documents:
            stmt = pg_insert(Document).values(
                url=doc.url,
                title=doc.title,
                raw_text=doc.raw_text,
                category=doc.category,
                content_hash=doc.content_hash,
                raw_html=doc.raw_html,
                document_kind=doc.document_kind,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["url"],
                set_={
                    "title": stmt.excluded.title,
                    "raw_text": stmt.excluded.raw_text,
                    "content_hash": stmt.excluded.content_hash,
                    "raw_html": stmt.excluded.raw_html,
                    "document_kind": stmt.excluded.document_kind,
                    "updated_at": func.now(),
                },
            )
            await self._session.execute(stmt)

    async def hash_map(self) -> dict[str, str]:
        stmt = select(Document.url, Document.content_hash)
        rows = (await self._session.execute(stmt)).all()
        return {row.url: row.content_hash for row in rows}

    async def known_urls(self) -> set[str]:
        stmt = select(Document.url)
        rows = (await self._session.execute(stmt)).scalars().all()
        return set(rows)

    async def scrape_status(self) -> ScrapeStatusRecord:
        recent_cutoff = func.now() - literal_column("INTERVAL '24 hours'")
        stmt = select(
            func.count().label("total"),
            func.count(case((Document.scraped_at > recent_cutoff, 1))).label("recent"),
            func.min(Document.scraped_at).label("first_scrape"),
            func.max(Document.scraped_at).label("last_scrape"),
        )
        row = (await self._session.execute(stmt)).one()
        return ScrapeStatusRecord(
            total=row.total,
            recent=row.recent,
            first_scrape=row.first_scrape,
            last_scrape=row.last_scrape,
        )

    async def pending_ids(self) -> list[UUID]:
        stmt = select(Document.id).where(
            or_(Document.ingested_hash.is_(None), Document.content_hash != Document.ingested_hash)
        )
        rows = (await self._session.execute(stmt)).scalars().all()
        return list(rows)

    async def all_ids(self) -> list[UUID]:
        stmt = select(Document.id)
        rows = (await self._session.execute(stmt)).scalars().all()
        return list(rows)

    async def mark_ingested(self, document_id: UUID, content_hash: str) -> None:
        stmt = update(Document).where(Document.id == document_id).values(ingested_hash=content_hash)
        await self._session.execute(stmt)

    async def commit(self) -> None:
        await self._session.commit()


async def get_scrape_status(session: AsyncSession) -> ScrapeStatusRecord:
    return await DocumentStore(session).scrape_status()


async def get_hash_map(session: AsyncSession) -> dict[str, str]:
    return await DocumentStore(session).hash_map()


async def save_documents(session: AsyncSession, documents: list[Document]) -> None:
    await DocumentStore(session).save(documents)


async def get_document_by_id(session: AsyncSession, doc_id: UUID) -> Document | None:
    return await DocumentStore(session).get(doc_id)


async def get_pending_document_ids(session: AsyncSession) -> list[UUID]:
    return await DocumentStore(session).pending_ids()


async def get_all_document_ids(session: AsyncSession) -> list[UUID]:
    return await DocumentStore(session).all_ids()


async def update_ingested_hash(session: AsyncSession, doc_id: UUID, hash_val: str) -> None:
    await DocumentStore(session).mark_ingested(doc_id, hash_val)


async def get_known_urls(session: AsyncSession) -> set[str]:
    return await DocumentStore(session).known_urls()
