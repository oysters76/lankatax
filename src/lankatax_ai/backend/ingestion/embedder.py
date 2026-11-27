import dataclasses
import functools
import logging
import time

from lankatax_ai.backend.ingestion.retry import retry_async
from lankatax_ai.backend.ingestion.types import Chunk, DocumentEmbeddings, EmbeddingPolicy

logger = logging.getLogger(__name__)


def _batches(chunks: list[Chunk], max_batch_tokens: int) -> list[list[Chunk]]:
    if max_batch_tokens < 1:
        raise ValueError("max_batch_tokens must be at least 1")

    batches: list[list[Chunk]] = []
    current: list[Chunk] = []
    current_tokens = 0

    for chunk in chunks:
        tokens = max(1, chunk.token_count)
        if current and current_tokens + tokens > max_batch_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(chunk)
        current_tokens += tokens

    if current:
        batches.append(current)
    return batches


async def embed_chunks(
    model: DocumentEmbeddings,
    chunks: list[Chunk],
    *,
    policy: EmbeddingPolicy,
) -> list[Chunk]:
    """Embed chunks in token-bounded batches, retrying transient failures."""
    if not chunks:
        return []

    batches = _batches(chunks, policy.max_batch_tokens)
    results: list[Chunk] = []

    for index, batch in enumerate(batches, start=1):
        texts = [chunk.content for chunk in batch]
        started = time.perf_counter()
        vectors = await retry_async(
            functools.partial(model.aembed_documents, texts),
            policy=policy.retry,
        )
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"Embedding provider returned {len(vectors)} vectors for {len(batch)} chunks"
            )

        results.extend(
            dataclasses.replace(chunk, embedding=vector)
            for chunk, vector in zip(batch, vectors, strict=True)
        )
        logger.info(
            "Embedded batch %d/%d (%d chunks) in %.2fs",
            index,
            len(batches),
            len(batch),
            time.perf_counter() - started,
        )

    return results
