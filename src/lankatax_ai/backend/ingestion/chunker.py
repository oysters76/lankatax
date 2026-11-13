from bs4 import BeautifulSoup as bs

from lankatax_ai.backend.ingestion.types import (
    BlockType,
    Chunk,
    ChunkCitationMeta,
    ChunkConfig,
    RawBlock,
)
from lankatax_ai.backend.scraper.pdf_extractor import PAGE_MARKER
from lankatax_ai.backend.scraper.storage import DocumentRecord
from lankatax_ai.backend.types import DocumentKind

RAW_BLOCK_MAP: dict[str, BlockType] = {
    "h1": BlockType.heading,
    "h2": BlockType.heading,
    "h3": BlockType.heading,
    "h4": BlockType.heading,
    "h5": BlockType.heading,
    "h6": BlockType.heading,
    "p": BlockType.paragraph,
    "table": BlockType.table,
    "ul": BlockType.unordered_list_block,
    "ol": BlockType.ordered_list_block,
    "caption": BlockType.caption,
}

ANCHOR_BLOCK_TYPES: set[BlockType] = {BlockType.heading}


def parse_html_to_blocks(raw_html: str) -> list[RawBlock]:
    soup = bs(raw_html, "html.parser")
    raw_blocks: list[RawBlock] = []
    for element in soup.find_all(RAW_BLOCK_MAP.keys()):
        text = element.get_text(strip=True)
        if not text:
            continue
        btype = RAW_BLOCK_MAP[element.name]
        raw_blocks.append(RawBlock(block_type=btype, content=text))
    return raw_blocks


def group_blocks(blocks: list[RawBlock]) -> list[list[RawBlock]]:
    groups: list[list[RawBlock]] = []
    group: list[RawBlock] = []
    for block in blocks:
        if block.block_type in ANCHOR_BLOCK_TYPES:
            if group:
                groups.append(group)
            group = [block]
        else:
            group.append(block)
    if len(group) > 0:
        groups.append(group)
    return groups


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


def _make_chunk(
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


def chunk_document(doc: DocumentRecord, config: ChunkConfig) -> list[Chunk]:
    if doc.document_kind == DocumentKind.page.value:
        texts = merge_blocks_to_chunks(group_blocks(parse_html_to_blocks(doc.raw_html)), config)
        return [_make_chunk(t, doc, config, i) for i, t in enumerate(texts)]
    elif doc.document_kind == DocumentKind.pdf.value:
        pages = doc.raw_text.split(PAGE_MARKER.split("{")[0])
        blocks = [
            RawBlock(block_type=BlockType.paragraph, content=p.strip()) for p in pages if p.strip()
        ]
        texts = merge_blocks_to_chunks([blocks], config)
        return [_make_chunk(t, doc, config, i) for i, t in enumerate(texts)]
    else:
        raise ValueError(f"Unknown document kind: {doc.document_kind}")
