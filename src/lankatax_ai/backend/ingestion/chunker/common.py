from lankatax_ai.backend.db.models import Document
from lankatax_ai.backend.documents.page_marker import (
    format_page_marker,
    page_numbers,
    strip_page_markers,
)
from lankatax_ai.backend.ingestion.types import (
    Chunk,
    ChunkCitationMeta,
    ChunkConfig,
    RawBlock,
)
from lankatax_ai.backend.types import DocumentKind


def token_size(s: str) -> int:
    return len(s.split())


def extract_pages_and_clean(text: str) -> tuple[int | None, int | None, str]:
    pages = page_numbers(text)
    page_start = pages[0] if pages else None
    page_end = pages[-1] if pages else None
    return page_start, page_end, strip_page_markers(text).strip()


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
                current = _overlap_with_page(overlap, current) + "\n" + block.content
    if current.strip():
        merged.append(current)
    return merged


def _overlap_with_page(overlap: str, current: str) -> str:
    """Keep the page marker of the page the overlap text came from.

    Chunks that start with overlap would otherwise only carry the *next* page's
    marker, so ``page_start`` would point one page too far.
    """
    if not overlap:
        return ""
    if page_numbers(overlap):
        return overlap
    current_pages = page_numbers(current)
    if not current_pages:
        return overlap
    return f"{format_page_marker(current_pages[-1])} {overlap}"


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
