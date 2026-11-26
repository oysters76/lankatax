import dataclasses
import logging
import time

from lankatax_ai.backend.ingestion.types import Chunk, DocumentEmbeddings

logger = logging.getLogger(__name__)


async def embed_chunks(model: DocumentEmbeddings, chunks: list[Chunk]) -> list[Chunk]:
    logger.debug("Embedding %d chunks", len(chunks))
    texts: list[str] = [c.content for c in chunks]
    t0 = time.perf_counter()
    vecs: list[list[float]] = await model.aembed_documents(texts)
    elapsed = time.perf_counter() - t0
    logger.info("Embedded %d chunks in %.2fs", len(chunks), elapsed)
    results: list[Chunk] = []
    for chunk, vec in zip(chunks, vecs):
        c_cloned = dataclasses.replace(chunk, embedding=vec)
        results.append(c_cloned)
    return results
