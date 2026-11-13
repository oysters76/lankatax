import dataclasses
import json
import uuid

import asyncpg

from lankatax_ai.backend.ingestion.types import Chunk


async def store_chunks(pool: asyncpg.Pool, document_id: uuid.UUID, chunks: list[Chunk]) -> None:
    query = """
        INSERT INTO chunks (document_id, content, embedding, chunk_index, metadata_json, token_count, chunk_strategy)
        VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
    """
    rows = [
        (
            document_id,
            c.content,
            str(c.embedding),
            i,
            json.dumps(dataclasses.asdict(c.citation)),
            c.token_count,
            c.chunk_strategy.value,
        )
        for i, c in enumerate(chunks)
    ]
    await pool.executemany(query, rows)


async def delete_chunks_by_url(pool: asyncpg.Pool, url: str) -> int:
    result = await pool.execute(
        "DELETE FROM chunks WHERE document_id = (SELECT id FROM documents WHERE url = $1)",
        url,
    )
    return int(result.split()[-1])


async def search_similar(
    pool: asyncpg.Pool, embedding: list[float], top_k: int = 20
) -> list[asyncpg.Record]:
    query = """
        SELECT id, document_id, content, metadata_json, token_count,
               embedding <=> $1 AS distance
        FROM chunks
        ORDER BY distance
        LIMIT $2
    """
    return await pool.fetch(query, str(embedding), top_k)
