import math
import re

from langchain_openai import OpenAIEmbeddings

from lankatax_ai.backend.db.models import Document
from lankatax_ai.backend.ingestion.chunker.common import extract_pages_and_clean, make_chunk
from lankatax_ai.backend.ingestion.types import Chunk, ChunkConfig


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
    embeddings: list[list[float]], threshold: float = 0.75, similarity=cosine_similarity
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
    if not sentences:
        return []
    embeddings: list[list[float]] = await model.aembed_documents(sentences)
    boundaries = find_boundaries(embeddings)
    chunks: list[Chunk] = []
    start = 0
    for boundary in boundaries:
        _append_chunk(sentences, start, boundary, doc, config, chunks)
        start = boundary
    _append_chunk(sentences, start, len(sentences), doc, config, chunks)
    return chunks
