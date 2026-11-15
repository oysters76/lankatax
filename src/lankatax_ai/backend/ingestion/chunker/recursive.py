import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from lankatax_ai.backend.db.models import Document
from lankatax_ai.backend.ingestion.chunker.common import extract_pages_and_clean, make_chunk
from lankatax_ai.backend.ingestion.types import Chunk, ChunkConfig

logger = logging.getLogger(__name__)


def chunk_recursive(doc: Document, config: ChunkConfig) -> list[Chunk]:
    logger.debug(
        "Recursive chunking %s — chunk_size=%d, overlap=%d",
        doc.url,
        config.max_tokens * 4,
        config.overlap_tokens * 4,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.max_tokens * 4,
        chunk_overlap=config.overlap_tokens * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    texts = splitter.split_text(doc.raw_text)
    logger.debug("Split into %d text segments", len(texts))
    chunks: list[Chunk] = []
    for i, t in enumerate(texts):
        page_start, page_end, cleaned = extract_pages_and_clean(t)
        chunks.append(make_chunk(cleaned, doc, config, i, page_start, page_end))
    return chunks
