import dataclasses
import enum
import json
import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from lankatax_ai.backend.ingestion.types import Chunk


class _EnumEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, enum.Enum):
            return o.value
        return super().default(o)


async def store_chunks(session: AsyncSession, document_id: uuid.UUID, chunks: list[Chunk]) -> None:
    query = text("""
        INSERT INTO chunks (document_id, content, embedding, chunk_index, metadata_json, token_count, chunk_strategy)
        VALUES (:document_id, :content, :embedding, :chunk_index, CAST(:metadata_json AS jsonb), :token_count, :chunk_strategy)
    """)
    for i, c in enumerate(chunks):
        await session.execute(
            query,
            {
                "document_id": document_id,
                "content": c.content,
                "embedding": c.embedding,
                "chunk_index": i,
                "metadata_json": json.dumps(dataclasses.asdict(c.citation), cls=_EnumEncoder),
                "token_count": c.token_count,
                "chunk_strategy": c.chunk_strategy.value,
            },
        )


async def delete_chunks_by_document_id(session: AsyncSession, document_id: uuid.UUID) -> int:
    result = await session.execute(
        text("DELETE FROM chunks WHERE document_id = :document_id"),
        {"document_id": document_id},
    )
    return result.rowcount


async def delete_chunks_by_url(session: AsyncSession, url: str) -> int:
    result = await session.execute(
        text("DELETE FROM chunks WHERE document_id = (SELECT id FROM documents WHERE url = :url)"),
        {"url": url},
    )
    return result.rowcount


async def search_similar(
    session: AsyncSession, embedding: list[float], top_k: int = 20
) -> list[dict]:
    query = text("""
        SELECT id, document_id, content, metadata_json, token_count,
               embedding <=> :embedding AS distance
        FROM chunks
        ORDER BY distance
        LIMIT :top_k
    """)
    result = await session.execute(query, {"embedding": embedding, "top_k": top_k})
    return [dict(row) for row in result.mappings().fetchall()]
