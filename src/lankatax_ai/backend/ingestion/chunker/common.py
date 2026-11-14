from lankatax_ai.backend.db.models import Document
from lankatax_ai.backend.ingestion.types import (
    Chunk,
    ChunkCitationMeta,
    ChunkConfig,
    RawBlock,
)
from lankatax_ai.backend.scraper.pdf_extractor import PAGE_MARKER_PATTERN
from lankatax_ai.backend.types import DocumentKind


def token_size(s: str) -> int:
    return len(s.split())


def extract_pages_and_clean(text: str) -> tuple[int | None, int | None, str]:
    pages = [int(p) for p in PAGE_MARKER_PATTERN.findall(text)]
    cleaned = PAGE_MARKER_PATTERN.sub("", text)
    page_start = pages[0] if pages else None
    page_end = pages[-1] if pages else None
    return page_start, page_end, cleaned.strip()


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
    content: str,
    doc: Document,
    config: ChunkConfig,
    order: int,
    page_start: int | None = None,
    page_end: int | None = None,
) -> Chunk:
    return Chunk(
        content=content,
        citation=ChunkCitationMeta(
            from_url=doc.url,
            document_title=doc.title,
            document_kind=DocumentKind(doc.document_kind),
            order=order,
            page_start=page_start,
            page_end=page_end,
        ),
        token_count=token_size(content),
        chunk_strategy=config.strategy,
    )
