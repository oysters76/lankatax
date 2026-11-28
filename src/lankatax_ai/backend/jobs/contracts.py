"""Workflow input and output contracts for the Hatchet adapter."""

from __future__ import annotations

from typing import Literal
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
    full: bool = False
    discover_pdfs: bool = True
    batch_size: int = Field(25, ge=1, le=100)


class IngestInput(BaseModel):
    target: Literal["pending", "all"] = "pending"
    max_documents: int | None = Field(None, ge=1)
    max_tokens: int | None = Field(None, ge=1)
    batch_size: int = Field(25, ge=1, le=100)


class IngestDocumentInput(BaseModel):
    document_id: UUID
    max_tokens: int | None = Field(None, ge=1)
    batch_size: int = Field(25, ge=1, le=100)


class CacheExpireInput(BaseModel):
    pass


class RefreshInput(BaseModel):
    full: bool = False
    discover_pdfs: bool = True
    target: Literal["pending", "all"] = "pending"
    max_documents: int | None = Field(None, ge=1)
    max_tokens: int | None = Field(None, ge=1)
    batch_size: int = Field(25, ge=1, le=100)


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
