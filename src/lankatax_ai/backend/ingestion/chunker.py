from bs4 import BeautifulSoup as bs

from lankatax_ai.backend.ingestion.types import BlockType, RawBlock

RAW_BLOCK_MAP = {
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
