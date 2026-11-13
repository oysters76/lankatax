from lankatax_ai.backend.ingestion.types import (
    Chunk,
    ChunkCitationMeta,
    ChunkConfig,
    RawBlock,
)
from lankatax_ai.backend.scraper.storage import DocumentRecord
from lankatax_ai.backend.types import DocumentKind


def token_size(s: str) -> int:
    return len(s.split())


def merge_blocks_to_chunks(groups: list[list[RawBlock]], config: ChunkConfig) -> list[str]:
    current = ""
    merged = []
    for group in groups:
        for block in group:
            current_len = token_size(current)
            is_over_limit = (current_len + token_size(block.content)) >= config.max_tokens
            if not is_over_limit:
                current += "\n" + block.content
            else:
                merged.append(current)
                overlap = " ".join(current.split()[-config.overlap_tokens :])
                current = overlap + "\n" + block.content
    if current.strip():
        merged.append(current)
    return merged


def make_chunk(
    content: str, doc: DocumentRecord, config: ChunkConfig, order: int, page: int | None = None
) -> Chunk:
    return Chunk(
        content=content,
        citation=ChunkCitationMeta(
            from_url=doc.url,
            document_title=doc.title,
            document_kind=DocumentKind(doc.document_kind),
            page_in_pdf=page,
            order=order,
        ),
        token_count=token_size(content),
        chunk_strategy=config.strategy,
    )
