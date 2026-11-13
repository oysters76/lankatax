from langchain_openai import OpenAIEmbeddings

from lankatax_ai.backend.ingestion.chunker.recursive import chunk_recursive
from lankatax_ai.backend.ingestion.chunker.semantic import chunk_semantic
from lankatax_ai.backend.ingestion.chunker.structural import chunk_structural
from lankatax_ai.backend.ingestion.types import Chunk, ChunkConfig, ChunkStrategy
from lankatax_ai.backend.scraper.storage import DocumentRecord


async def chunk_document(
    doc: DocumentRecord, config: ChunkConfig, model: OpenAIEmbeddings | None = None
) -> list[Chunk]:
    if config.strategy == ChunkStrategy.structural:
        return chunk_structural(doc, config)
    elif config.strategy == ChunkStrategy.semantic:
        if model is None:
            raise ValueError("Semantic chunking requires an OpenAIEmbeddings model")
        return await chunk_semantic(doc, config, model)
    elif config.strategy == ChunkStrategy.recursive:
        return chunk_recursive(doc, config)
    else:
        raise ValueError(f"Unknown chunk strategy: {config.strategy}")
