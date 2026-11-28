"""HTTP schemas: hypermedia envelope, requests, responses and errors.

Every response model inherits :class:`Hypermedia`, so a client can navigate the
API by following ``links`` instead of hardcoding URLs (Richardson Level 3).
These models are presentation only; they are mapped to and from service types
in ``utils.py``.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from lankatax_ai.backend.services.jobs import JobStatus, JobType, JsonObject
from lankatax_ai.backend.services.search import limits
from lankatax_ai.backend.types import LlmProvider, QueryType, SearchMode

__all__ = [
    "ApiRoot",
    "CacheResponse",
    "CountResponse",
    "ErrorBody",
    "Health",
    "Hypermedia",
    "JobActionSchema",
    "JobDetailsSchema",
    "JobListSchema",
    "JobReceiptSchema",
    "JobRequestSchema",
    "JobSummarySchema",
    "JobTaskSchema",
    "Link",
    "SearchRequest",
    "SearchResponse",
    "SearchResultItem",
    "ServiceRoot",
    "StatsResponse",
    "TimingModel",
    "TokenResponse",
]

MAX_QUERY_CHARS = 2000


class Link(BaseModel):
    """One action a client may take next."""

    href: str
    method: str = "GET"
    title: str | None = None
    available: bool = True
    auth_required: bool = False


class Hypermedia(BaseModel):
    """Base for every response; ``links`` is the navigation surface."""

    links: dict[str, Link] = Field(default_factory=dict)


class ServiceRoot(Hypermedia):
    name: str
    version: str
    description: str


class Health(Hypermedia):
    status: Literal["ok"] = "ok"
    database: Literal["ok"] = "ok"


class ApiRoot(Hypermedia):
    version: str


class TokenResponse(Hypermedia):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int


class ErrorBody(BaseModel):
    """The single error shape of the API; mirrors the ``links`` contract."""

    code: str
    detail: str
    errors: list[dict[str, Any]] | None = None
    links: dict[str, Link] = Field(default_factory=dict)


class CountResponse(Hypermedia):
    """Result of a bulk cache mutation."""

    deleted: int


class SearchRequest(BaseModel):
    """Search parameters; bounds come from the service so both agree."""

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    mode: SearchMode = SearchMode.hybrid
    top_k: int = Field(limits.DEFAULT_TOP_K, ge=limits.MIN_TOP_K, le=limits.MAX_TOP_K)
    rerank: bool = False
    multi_query: bool = False
    use_cache: bool = True
    include_context: bool = False
    vector_weight: float = Field(
        limits.DEFAULT_VECTOR_WEIGHT,
        ge=limits.MIN_VECTOR_WEIGHT,
        le=limits.MAX_VECTOR_WEIGHT,
    )
    rerank_threshold: int = Field(
        limits.DEFAULT_RERANK_THRESHOLD,
        ge=limits.MIN_RERANK_THRESHOLD,
        le=limits.MAX_RERANK_THRESHOLD,
    )
    dedup_threshold: float = Field(
        limits.DEFAULT_DEDUP_THRESHOLD,
        ge=limits.MIN_DEDUP_THRESHOLD,
        le=limits.MAX_DEDUP_THRESHOLD,
    )
    token_budget: int | None = Field(None, ge=limits.MIN_TOKEN_BUDGET, le=limits.MAX_TOKEN_BUDGET)
    llm_provider: LlmProvider = LlmProvider.gemini
    category: str | None = None
    doc_type: str | None = None
    after: date | None = None
    before: date | None = None


class TimingModel(BaseModel):
    embed_ms: int
    search_ms: int
    rerank_ms: int
    total_ms: int


class SearchResultItem(Hypermedia):
    """One retrieved chunk.

    ``embedding`` is deliberately absent: it is an internal detail and would
    bloat every response.
    """

    chunk_id: UUID
    document_id: UUID
    content: str
    score: float
    vector_score: float | None = None
    keyword_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(Hypermedia):
    query: str
    expanded_query: str
    query_type: QueryType
    results: list[SearchResultItem]
    timings: TimingModel
    cache_hit: bool
    multi_query: bool
    context: str | None = None


class StatsResponse(Hypermedia):
    total_queries: int
    avg_latency_ms: float
    hit_rate: float
    no_result_queries: list[str]
    avg_top_score: float
    cache_hit_count: int
    multi_query_count: int


class CacheResponse(Hypermedia):
    total_entries: int
    total_hits: int
    hit_rate: float
    oldest_entry_hours: float


class JobRequestSchema(BaseModel):
    type: JobType
    params: JsonObject = Field(default_factory=dict)
    idempotency_key: str | None = Field(None, max_length=200)


class JobTaskSchema(Hypermedia):
    id: str
    name: str
    status: JobStatus
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: int | None = None
    output: JsonObject | None = None
    error: str | None = None


class JobSummarySchema(Hypermedia):
    id: str
    type: JobType
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: int | None = None
    requested_by: str | None = None
    error: str | None = None


class JobDetailsSchema(JobSummarySchema):
    input: JsonObject = Field(default_factory=dict)
    output: JsonObject | None = None
    tasks: list[JobTaskSchema] = Field(default_factory=list)


class JobReceiptSchema(Hypermedia):
    run_id: str
    reused: bool


class JobActionSchema(Hypermedia):
    run_id: str
    action: Literal["cancel", "replay"]


class JobListSchema(Hypermedia):
    items: list[JobSummarySchema]
    next: Link | None = None
