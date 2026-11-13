from lankatax_ai.backend.ingestion.chunker.structural import chunk_structural
from lankatax_ai.backend.ingestion.types import Chunk, ChunkConfig, ChunkStrategy
from lankatax_ai.backend.scraper.storage import DocumentRecord


def chunk_document(doc: DocumentRecord, config: ChunkConfig) -> list[Chunk]:
    if config.strategy == ChunkStrategy.structural:
        return chunk_structural(doc, config)
    elif config.strategy == ChunkStrategy.semantic:
        raise NotImplementedError("Semantic chunking not yet implemented")
    else:
        raise ValueError(f"Unknown chunk strategy: {config.strategy}")
