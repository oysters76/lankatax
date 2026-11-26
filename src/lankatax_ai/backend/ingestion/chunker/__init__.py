import logging

from langchain_openai import OpenAIEmbeddings

from lankatax_ai.backend.db.models import Document
from lankatax_ai.backend.ingestion.chunker.recursive import chunk_recursive
from lankatax_ai.backend.ingestion.chunker.semantic import chunk_semantic
from lankatax_ai.backend.ingestion.chunker.structural import chunk_structural
from lankatax_ai.backend.ingestion.types import Chunk, ChunkConfig
from lankatax_ai.backend.types import ChunkStrategy

logger = logging.getLogger(__name__)


async def chunk_document(
    doc: Document, config: ChunkConfig, model: OpenAIEmbeddings | None = None
) -> list[Chunk]:
    logger.info("Chunking doc %s with strategy=%s", doc.url, config.strategy.value)
    if config.strategy == ChunkStrategy.structural:
        chunks = chunk_structural(doc, config)
    elif config.strategy == ChunkStrategy.semantic:
        if model is None:
            raise ValueError("Semantic chunking requires an OpenAIEmbeddings model")
        chunks = await chunk_semantic(doc, config, model)
    elif config.strategy == ChunkStrategy.recursive:
        chunks = chunk_recursive(doc, config)
    else:
        raise ValueError(f"Unknown chunk strategy: {config.strategy}")
    logger.info("Produced %d chunks for %s", len(chunks), doc.url)
    return chunks
