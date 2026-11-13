import dataclasses
from datetime import datetime
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclasses.dataclass
class ScrapeStatusRecord:
    total: int
    recent: int
    first_scrape: datetime
    last_scrape: datetime


@dataclasses.dataclass
class ScrapeContentHashRecord:
    url: str
    content_hash: str


@dataclasses.dataclass
class DocumentRecord:
    url: str
    title: str
    raw_text: str
    category: str
    content_hash: str
    id: UUID | None = None
    raw_html: str = ""
    document_kind: str = "page"
    ingested_hash: str | None = None


async def get_scrape_status(session: AsyncSession) -> ScrapeStatusRecord:
    query = """SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN scraped_at > NOW() - INTERVAL '24 hours' THEN 1 END) as recent,
            MIN(scraped_at) as first_scrape,
            MAX(scraped_at) as last_scrape
        FROM documents;"""
    result = await session.execute(text(query))
    row = result.mappings().fetchone()
    return ScrapeStatusRecord(**row)


async def get_hash_map(session: AsyncSession) -> dict[str, ScrapeContentHashRecord]:
    query = """
            SELECT url, content_hash
            FROM documents
        """
    result = await session.execute(text(query))
    return {row["url"]: ScrapeContentHashRecord(**row) for row in result.mappings()}


async def save_documents(session: AsyncSession, document_records: list[DocumentRecord]):
    params = [dataclasses.asdict(d) for d in document_records]
    query = """
        INSERT INTO documents (url, title, raw_text, category, content_hash, raw_html, document_kind)
        VALUES (:url, :title, :raw_text, :category, :content_hash, :raw_html, :document_kind)
        ON CONFLICT (url) DO UPDATE SET
            title = :title,
            raw_text = :raw_text,
            content_hash = :content_hash,
            raw_html = :raw_html,
            document_kind = :document_kind,
            updated_at = NOW()
    """
    await session.execute(text(query), params)
    await session.commit()


async def get_document_by_id(session: AsyncSession, doc_id: UUID) -> DocumentRecord | None:
    query = """
        SELECT id, url, title, raw_text, category, content_hash, raw_html, document_kind, ingested_hash
        FROM documents where id = :doc_id 
    """
    params = {"doc_id": doc_id}
    results = await session.execute(text(query), params)
    row = results.mappings().fetchone()
    return None if row is None else DocumentRecord(**row)


async def get_pending_document_ids(session: AsyncSession) -> list[UUID]:
    query = """
    SELECT id FROM documents
    WHERE ingested_hash IS NULL OR content_hash != ingested_hash 
    """
    results = await session.execute(text(query))
    return [row["id"] for row in results.mappings().fetchall()]


async def get_all_document_ids(session: AsyncSession) -> list[UUID]:
    query = """
    SELECT id FROM documents 
    """
    results = await session.execute(text(query))
    return [row["id"] for row in results.mappings().fetchall()]


async def update_ingested_hash(session: AsyncSession, doc_id: UUID, hash_val: str) -> None:
    params = {"doc_id": doc_id, "hash_val": hash_val}
    query = """
    UPDATE documents SET ingested_hash = :hash_val WHERE id = :doc_id 
    """
    await session.execute(text(query), params)
    await session.commit()


async def get_known_urls(session: AsyncSession) -> set[str]:
    query = """
    SELECT url FROM documents
    """
    results = await session.execute(text(query))
    return {row["url"] for row in results.mappings().fetchall()}
