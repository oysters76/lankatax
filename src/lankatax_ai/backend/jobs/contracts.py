"""Workflow input and output contracts for the Hatchet adapter."""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field

__all__ = [
    "CacheExpireInput",
    "CacheOutput",
    "IngestDocumentInput",
    "IngestInput",
    "IngestOutput",
    "RefreshInput",
    "ScrapeInput",
    "ScrapeOutput",
]


class ScrapeInput(BaseModel):
    idempotency_key: str = "scrape-registry"
    full: bool = False
    discover_pdfs: bool = True
    batch_size: Annotated[int, Field(ge=1, le=100)] = 25


class IngestInput(BaseModel):
    idempotency_key: str = "ingest-pending"
    target: Literal["pending", "all"] = "pending"
    max_documents: Annotated[int | None, Field(ge=1)] = None
    max_tokens: Annotated[int | None, Field(ge=1)] = None
    batch_size: Annotated[int, Field(ge=1, le=100)] = 25


class IngestDocumentInput(BaseModel):
    document_id: UUID
    max_tokens: Annotated[int | None, Field(ge=1)] = None
    batch_size: Annotated[int, Field(ge=1, le=100)] = 25


class CacheExpireInput(BaseModel):
    pass


class RefreshInput(BaseModel):
    idempotency_key: str = "pipeline-refresh"
    full: bool = False
    discover_pdfs: bool = True
    target: Literal["pending", "all"] = "pending"
    max_documents: Annotated[int | None, Field(ge=1)] = None
    max_tokens: Annotated[int | None, Field(ge=1)] = None
    batch_size: Annotated[int, Field(ge=1, le=100)] = 25


class ScrapeOutput(BaseModel):
    saved: int
    skipped: int
    failed: int
    discovered: int
    partial: bool
    errors: list[str] = Field(default_factory=list)
    duration_ms: int


class IngestOutput(BaseModel):
    docs_processed: int
    chunks_created: int
    tokens_used: int
    partial: bool
    budget_exhausted: bool = False
    errors: list[str] = Field(default_factory=list)
    duration_ms: int


class CacheOutput(BaseModel):
    expired: int
    duration_ms: int
