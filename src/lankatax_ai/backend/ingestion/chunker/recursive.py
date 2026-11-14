from langchain_text_splitters import RecursiveCharacterTextSplitter

from lankatax_ai.backend.db.models import Document
from lankatax_ai.backend.ingestion.chunker.common import extract_pages_and_clean, make_chunk
from lankatax_ai.backend.ingestion.types import Chunk, ChunkConfig


def chunk_recursive(doc: Document, config: ChunkConfig) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.max_tokens * 4,
        chunk_overlap=config.overlap_tokens * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    texts = splitter.split_text(doc.raw_text)
    chunks: list[Chunk] = []
    for i, t in enumerate(texts):
        page_start, page_end, cleaned = extract_pages_and_clean(t)
        chunks.append(make_chunk(cleaned, doc, config, i, page_start, page_end))
    return chunks
