import dataclasses
import json
import uuid

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from lankatax_ai.backend.db.models import Chunk as ChunkModel
from lankatax_ai.backend.db.models import Document
from lankatax_ai.backend.ingestion.types import Chunk
from lankatax_ai.backend.utils import EnumEncoder


async def store_chunks(session: AsyncSession, document_id: uuid.UUID, chunks: list[Chunk]) -> None:
    models = [
        ChunkModel(
            document_id=document_id,
            content=c.content,
            embedding=c.embedding,
            chunk_index=i,
            metadata_json=json.loads(json.dumps(dataclasses.asdict(c.citation), cls=EnumEncoder)),
            token_count=c.token_count,
            chunk_strategy=c.chunk_strategy.value,
        )
        for i, c in enumerate(chunks)
    ]
    session.add_all(models)


async def delete_chunks_by_document_id(session: AsyncSession, document_id: uuid.UUID) -> int:
    result = await session.execute(delete(ChunkModel).where(ChunkModel.document_id == document_id))
    return result.rowcount


async def delete_chunks_by_url(session: AsyncSession, url: str) -> int:
    subq = select(Document.id).where(Document.url == url).scalar_subquery()
    result = await session.execute(delete(ChunkModel).where(ChunkModel.document_id == subq))
    return result.rowcount


async def search_similar(
    session: AsyncSession, embedding: list[float], top_k: int = 20
) -> list[dict]:
    distance = ChunkModel.embedding.cosine_distance(embedding).label("distance")
    stmt = (
        select(
            ChunkModel.id,
            ChunkModel.document_id,
            ChunkModel.content,
            ChunkModel.metadata_json,
            ChunkModel.token_count,
            distance,
        )
        .order_by(distance)
        .limit(top_k)
    )
    rows = (await session.execute(stmt)).mappings().all()
    return [dict(row) for row in rows]
