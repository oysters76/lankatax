import dataclasses
from datetime import datetime

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
        INSERT INTO documents (url, title, raw_text, category, content_hash)
        VALUES (:url, :title, :raw_text, :category, :content_hash)
        ON CONFLICT (url) DO UPDATE SET
            title = :title,
            raw_text = :raw_text,
            content_hash = :content_hash,
            updated_at = NOW()
    """
    await session.execute(text(query), params)
    await session.commit()
