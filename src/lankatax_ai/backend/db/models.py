from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, CheckConstraint, ForeignKey, Index, Integer, Numeric, Text, text
from sqlalchemy.dialects.postgresql import ENUM as PgEnum
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from lankatax_ai.backend.types import (
    ChunkStrategy,
    DocumentKind,
    LlmProvider,
    QueryType,
    SearchMode,
    SearchSource,
)


class Base(DeclarativeBase):
    pass


def _pg_enum(name: str, python_enum: type[Enum]) -> PgEnum:
    """Build a Postgres enum whose values are derived from a Python enum.

    Keeping one source of truth stops the database type from drifting away from
    the application vocabulary.
    """
    return PgEnum(*(member.value for member in python_enum), name=name, create_type=True)


document_kind_enum = _pg_enum("document_kind_enum", DocumentKind)
chunk_strategy_enum = _pg_enum("chunk_strategy_enum", ChunkStrategy)
search_source_enum = _pg_enum("search_source_enum", SearchSource)
search_mode_enum = _pg_enum("search_mode_enum", SearchMode)
query_type_enum = _pg_enum("query_type_enum", QueryType)
llm_provider_enum = _pg_enum("llm_provider_enum", LlmProvider)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(
        primary_key=True, default=uuid4, server_default=text("gen_random_uuid()")
    )
    url: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(Text, nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    raw_html: Mapped[str | None] = mapped_column(Text, nullable=True)
    document_kind: Mapped[str] = mapped_column(
        document_kind_enum, nullable=False, server_default="page"
    )
    ingested_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    scraped_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default="now()")
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default="now()")

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        Index(
            "idx_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index(
            "idx_chunks_content_trgm",
            "content",
            postgresql_using="gin",
            postgresql_ops={"content": "gin_trgm_ops"},
        ),
    )

    id: Mapped[UUID] = mapped_column(
        primary_key=True, default=uuid4, server_default=text("gen_random_uuid()")
    )
    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding = mapped_column(Vector(1536))
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_json = mapped_column(JSONB, server_default="{}")
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_strategy: Mapped[str | None] = mapped_column(chunk_strategy_enum, nullable=True)

    document: Mapped[Document] = relationship(back_populates="chunks")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[UUID] = mapped_column(
        primary_key=True, default=uuid4, server_default=text("gen_random_uuid()")
    )
    fingerprint: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default="now()")
    message_count: Mapped[int] = mapped_column(Integer, server_default="0")

    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="ck_chat_messages_role"),
    )

    id: Mapped[UUID] = mapped_column(
        primary_key=True, default=uuid4, server_default=text("gen_random_uuid()")
    )
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources_json = mapped_column(JSONB, server_default="[]")
    tokens_used: Mapped[int] = mapped_column(Integer, server_default="0")
    cost: Mapped[Decimal] = mapped_column(Numeric(10, 6), server_default="0")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default="now()")

    session: Mapped[ChatSession] = relationship(back_populates="messages")


class CorpusState(Base):
    """Singleton row that versions the searchable corpus.

    A successful ingest advances ``epoch``. Cache reads are scoped to the
    current epoch, so readers stop seeing results produced by an older corpus
    without requiring a destructive cache flush.
    """

    __tablename__ = "corpus_state"
    __table_args__ = (CheckConstraint("id = 1", name="ck_corpus_state_singleton"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1, server_default="1")
    epoch: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    last_ingest_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    last_refresh_run_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default="now()")


class EvalResult(Base):
    __tablename__ = "eval_results"
    __table_args__ = (
        CheckConstraint("faithfulness BETWEEN 0 AND 1", name="ck_eval_results_faithfulness"),
        CheckConstraint("relevance BETWEEN 0 AND 1", name="ck_eval_results_relevance"),
    )

    id: Mapped[UUID] = mapped_column(
        primary_key=True, default=uuid4, server_default=text("gen_random_uuid()")
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str] = mapped_column(Text, nullable=False)
    actual_answer: Mapped[str] = mapped_column(Text, nullable=False)
    faithfulness: Mapped[Decimal | None] = mapped_column(Numeric(3, 2), nullable=True)
    relevance: Mapped[Decimal | None] = mapped_column(Numeric(3, 2), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default="now()")


class CacheEntry(Base):
    __tablename__ = "cache_entries"
    __table_args__ = (
        Index(
            "idx_cache_embedding",
            "query_embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"query_embedding": "vector_cosine_ops"},
        ),
        Index("idx_cache_entries_epoch_created", "corpus_epoch", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(
        primary_key=True, default=uuid4, server_default=text("gen_random_uuid()")
    )
    query_embedding = mapped_column(Vector(1536))
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[str] = mapped_column(Text, nullable=False)
    corpus_epoch: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default="now()")
    hit_count: Mapped[int] = mapped_column(Integer, server_default="0")


class SearchLog(Base):
    __tablename__ = "search_logs"

    id: Mapped[UUID] = mapped_column(
        primary_key=True, default=uuid4, server_default=text("gen_random_uuid()")
    )
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    expanded_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    query_type: Mapped[str] = mapped_column(query_type_enum, nullable=False)
    search_mode: Mapped[str] = mapped_column(search_mode_enum, nullable=False)
    top_score: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)
    result_count: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_embed_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_search_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_rerank_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_total_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    llm_provider: Mapped[str | None] = mapped_column(llm_provider_enum, nullable=True)
    cache_hit: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    multi_query: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    source: Mapped[str] = mapped_column(search_source_enum, nullable=False, server_default="'cli'")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default="now()")
