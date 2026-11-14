import dataclasses
from datetime import datetime
from uuid import UUID

from sqlalchemy import case, func, literal_column, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from lankatax_ai.backend.db.models import Document


@dataclasses.dataclass
class ScrapeStatusRecord:
    total: int
    recent: int
    first_scrape: datetime
    last_scrape: datetime


async def get_scrape_status(session: AsyncSession) -> ScrapeStatusRecord:
    recent_cutoff = func.now() - literal_column("INTERVAL '24 hours'")
    stmt = select(
        func.count().label("total"),
        func.count(case((Document.scraped_at > recent_cutoff, 1))).label("recent"),
        func.min(Document.scraped_at).label("first_scrape"),
        func.max(Document.scraped_at).label("last_scrape"),
    )
    row = (await session.execute(stmt)).one()
    return ScrapeStatusRecord(
        total=row.total,
        recent=row.recent,
        first_scrape=row.first_scrape,
        last_scrape=row.last_scrape,
    )


async def get_hash_map(session: AsyncSession) -> dict[str, str]:
    stmt = select(Document.url, Document.content_hash)
    rows = (await session.execute(stmt)).all()
    return {row.url: row.content_hash for row in rows}


async def save_documents(session: AsyncSession, documents: list[Document]):
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
        await session.execute(stmt)
    await session.commit()


async def get_document_by_id(session: AsyncSession, doc_id: UUID) -> Document | None:
    return await session.get(Document, doc_id)


async def get_pending_document_ids(session: AsyncSession) -> list[UUID]:
    stmt = select(Document.id).where(
        or_(Document.ingested_hash.is_(None), Document.content_hash != Document.ingested_hash)
    )
    rows = (await session.execute(stmt)).scalars().all()
    return list(rows)


async def get_all_document_ids(session: AsyncSession) -> list[UUID]:
    stmt = select(Document.id)
    rows = (await session.execute(stmt)).scalars().all()
    return list(rows)


async def update_ingested_hash(session: AsyncSession, doc_id: UUID, hash_val: str) -> None:
    stmt = update(Document).where(Document.id == doc_id).values(ingested_hash=hash_val)
    await session.execute(stmt)
    await session.commit()


async def get_known_urls(session: AsyncSession) -> set[str]:
    stmt = select(Document.url)
    rows = (await session.execute(stmt)).scalars().all()
    return set(rows)
