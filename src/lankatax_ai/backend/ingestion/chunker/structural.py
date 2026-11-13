from bs4 import BeautifulSoup as bs

from lankatax_ai.backend.ingestion.chunker.common import (
    extract_pages_and_clean,
    make_chunk,
    merge_blocks_to_chunks,
)
from lankatax_ai.backend.ingestion.types import (
    BlockType,
    Chunk,
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


def chunk_structural(doc: DocumentRecord, config: ChunkConfig) -> list[Chunk]:
    if doc.document_kind == DocumentKind.page.value:
        texts = merge_blocks_to_chunks(group_blocks(parse_html_to_blocks(doc.raw_html)), config)
        return [make_chunk(t, doc, config, i) for i, t in enumerate(texts)]
    elif doc.document_kind == DocumentKind.pdf.value:
        pages = doc.raw_text.split(PAGE_MARKER.split("{")[0])
        blocks = [
            RawBlock(block_type=BlockType.paragraph, content=p.strip()) for p in pages if p.strip()
        ]
        texts = merge_blocks_to_chunks([blocks], config)
        chunks = []
        for i, t in enumerate(texts):
            page_start, page_end, cleaned = extract_pages_and_clean(t)
            chunks.append(make_chunk(cleaned, doc, config, i, page_start, page_end))
        return chunks
    else:
        raise ValueError(f"Unknown document kind: {doc.document_kind}")
