import dataclasses
from collections.abc import Callable
from enum import Enum
from typing import Protocol
from uuid import UUID

from lankatax_ai.backend.db.models import Document
from lankatax_ai.backend.types import ChunkStrategy, DocumentKind


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
    order: int = 0
    page_start: int | None = None
    page_end: int | None = None


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


@dataclasses.dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_retries: int = 3
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 10.0
    jitter_ratio: float = 0.2

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if self.backoff_base_seconds < 0 or self.backoff_max_seconds < 0:
            raise ValueError("backoff durations must not be negative")
        if not 0 <= self.jitter_ratio <= 1:
            raise ValueError("jitter_ratio must be between 0 and 1")


@dataclasses.dataclass(frozen=True, slots=True)
class EmbeddingPolicy:
    max_batch_tokens: int = 8000
    retry: RetryPolicy = dataclasses.field(default_factory=RetryPolicy)

    def __post_init__(self) -> None:
        if self.max_batch_tokens < 1:
            raise ValueError("max_batch_tokens must be at least 1")


@dataclasses.dataclass(slots=True)
class IngestionResult:
    docs_processed: int = 0
    chunks_created: int = 0
    errors: list[str] = dataclasses.field(default_factory=list)
    tokens_used: int = 0
    budget_exhausted: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class IngestBatch:
    """The result of committing, or failing, one document."""

    document_id: UUID
    title: str
    chunks_created: int
    tokens_used: int
    error: str | None = None
    budget_exhausted: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class IngestionStatus:
    total_documents: int
    pending: int
    ingested: int


class DocumentEmbeddings(Protocol):
    """The slice of an embeddings client the ingestion pipeline needs."""

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]: ...


# Called after each document: (document id, chunks, title, tokens, error).
ProgressCallback = Callable[[UUID, int, str, int, str | None], None]


class IngestionRepository(Protocol):
    """Persistence the ingestion pipeline needs.

    The repository is also the unit of work: the orchestrator commits after each
    document and rolls back a failed one through it, so the same flow works
    whether the caller passes a request-scoped session or a fresh one.
    """

    async def get_document(self, document_id: UUID) -> Document | None: ...

    async def pending_document_ids(self) -> list[UUID]: ...

    async def all_document_ids(self) -> list[UUID]: ...

    async def status(self) -> IngestionStatus: ...

    async def replace_chunks(self, document_id: UUID, chunks: list[Chunk]) -> int: ...

    async def mark_ingested(self, document_id: UUID, content_hash: str) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...
