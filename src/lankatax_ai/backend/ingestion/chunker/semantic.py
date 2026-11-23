import logging
import math
import re
import time
from collections.abc import Callable

from langchain_openai import OpenAIEmbeddings

from lankatax_ai.backend.db.models import Document
from lankatax_ai.backend.ingestion.chunker.common import extract_pages_and_clean, make_chunk
from lankatax_ai.backend.ingestion.types import Chunk, ChunkConfig

logger = logging.getLogger(__name__)


def dot(a: list[float], b: list[float]) -> float:
    return sum(i * j for i, j in zip(a, b))


def absolute(a: list[float]) -> float:
    return sum(i**2 for i in a)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    ab = math.sqrt(absolute(a) * absolute(b))
    if ab == 0:
        return 0
    d = dot(a, b)
    return d / ab


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def find_boundaries(
    embeddings: list[list[float]],
    threshold: float = 0.75,
    similarity: Callable[[list[float], list[float]], float] = cosine_similarity,
) -> list[int]:
    boundaries = []
    for i in range(len(embeddings) - 1):
        sim = similarity(embeddings[i], embeddings[i + 1])
        if sim < threshold:
            boundaries.append(i + 1)
    return boundaries


def _append_chunk(
    sentences: list[str],
    start: int,
    end: int,
    doc: Document,
    config: ChunkConfig,
    chunks: list[Chunk],
) -> None:
    raw = " ".join(sentences[start:end])
    if not raw.strip():
        return
    page_start, page_end, cleaned = extract_pages_and_clean(raw)
    chunks.append(make_chunk(cleaned, doc, config, len(chunks), page_start, page_end))


async def chunk_semantic(
    doc: Document, config: ChunkConfig, model: OpenAIEmbeddings
) -> list[Chunk]:
    sentences = split_sentences(doc.raw_text)
    logger.debug("Semantic chunking %s — %d sentences", doc.url, len(sentences))
    if not sentences:
        logger.warning("No sentences found for %s, skipping", doc.url)
        return []
    t0 = time.perf_counter()
    embeddings: list[list[float]] = await model.aembed_documents(sentences)
    elapsed = time.perf_counter() - t0
    logger.debug("Sentence embeddings for %s took %.2fs", doc.url, elapsed)
    boundaries = find_boundaries(embeddings)
    logger.debug("Found %d semantic boundaries", len(boundaries))
    chunks: list[Chunk] = []
    start = 0
    for boundary in boundaries:
        _append_chunk(sentences, start, boundary, doc, config, chunks)
        start = boundary
    _append_chunk(sentences, start, len(sentences), doc, config, chunks)
    return chunks
