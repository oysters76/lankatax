"""Workflow input and output contracts for the Hatchet adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from lankatax_ai.backend.services.jobs.types import JobType

__all__ = [
    "JOB_INPUTS",
    "CacheExpireInput",
    "CacheOutput",
    "IngestDocumentInput",
    "IngestInput",
    "IngestOutput",
    "RefreshInput",
    "ScrapeInput",
    "ScrapeOutput",
    "serialize_input",
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


#: Workflow input model per job type, used to validate and default engine inputs.
JOB_INPUTS: dict[JobType, type[BaseModel]] = {
    JobType.pipeline_refresh: RefreshInput,
    JobType.scrape_registry: ScrapeInput,
    JobType.ingest_pending: IngestInput,
    JobType.ingest_document: IngestDocumentInput,
    JobType.cache_expire: CacheExpireInput,
}


def serialize_input(job_type: JobType, params: Mapping[str, Any]) -> dict[str, Any]:
    """Validate params against the workflow contract, applying its defaults.

    Hatchet evaluates idempotency against the raw input it receives, so defaults
    such as ``idempotency_key`` must be present in the payload, not only in the
    task signature.
    """
    model = JOB_INPUTS[job_type]
    return model.model_validate(dict(params)).model_dump(mode="json")
