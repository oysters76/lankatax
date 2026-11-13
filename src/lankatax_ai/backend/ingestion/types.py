import dataclasses
from enum import Enum

from lankatax_ai.backend.types import DocumentKind


class ChunkStrategy(Enum):
    structural = "structural"
    semantic = "semantic"


class BlockType(Enum):
    heading = "heading"
    paragraph = "paragraph"
    table = "table"
    ordered_list_block = "olist"
    unordered_list_block = "ulist"
    caption = "caption"


@dataclasses.dataclass
class ChunkCitationMeta:
    from_url: str
    document_title: str
    document_kind: DocumentKind
    page_in_pdf: int | None = None
    order: int = 0


@dataclasses.dataclass
class RawBlock:
    block_type: BlockType
    content: str


@dataclasses.dataclass
class Chunk:
    content: str
    citation: ChunkCitationMeta
    token_count: int
    chunk_strategy: ChunkStrategy
    embedding: list[float] | None = None


@dataclasses.dataclass
class ChunkConfig:
    max_tokens: int = 512
    overlap_tokens: int = 75
    strategy: ChunkStrategy = ChunkStrategy.structural
