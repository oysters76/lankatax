"""Job domain types shared by the API, services and job adapter."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from datetime import datetime
from enum import Enum

from lankatax_ai.backend.services.errors import ValidationError
from lankatax_ai.backend.services.jobs.limits import DEFAULT_LIMIT, MAX_LIMIT, MIN_LIMIT
from lankatax_ai.backend.services.utils import require_range

__all__ = [
    "JobDetails",
    "JobFilter",
    "JobRequest",
    "JobStatus",
    "JobSummary",
    "JobTask",
    "JobType",
    "JsonObject",
    "JsonValue",
]

type JsonValue = str | int | float | bool | None | list["JsonValue"] | Mapping[str, "JsonValue"]
type JsonObject = Mapping[str, JsonValue]


class JobType(Enum):
    pipeline_refresh = "pipeline.refresh"
    scrape_registry = "scrape.registry"
    ingest_pending = "ingest.pending"
    ingest_document = "ingest.document"
    cache_expire = "cache.expire"


class JobStatus(Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    succeeded_with_errors = "succeeded_with_errors"
    failed = "failed"
    cancelled = "cancelled"


@dataclasses.dataclass(frozen=True, slots=True)
class JobRequest:
    """A request to enqueue one job."""

    type: JobType
    params: JsonObject = dataclasses.field(default_factory=dict)
    idempotency_key: str | None = None
    requested_by: str | None = None

    def __post_init__(self) -> None:
        if self.idempotency_key is not None and not self.idempotency_key.strip():
            raise ValidationError("idempotency_key must not be blank")


@dataclasses.dataclass(frozen=True, slots=True)
class JobFilter:
    """Filtering and pagination for job listings."""

    statuses: tuple[JobStatus, ...] = ()
    types: tuple[JobType, ...] = ()
    since: datetime | None = None
    until: datetime | None = None
    limit: int = DEFAULT_LIMIT
    offset: int = 0

    def __post_init__(self) -> None:
        require_range("Limit", self.limit, MIN_LIMIT, MAX_LIMIT)
        if self.offset < 0:
            raise ValidationError("Offset must not be negative")
        if self.since is not None and self.until is not None and self.since > self.until:
            raise ValidationError("since must not be after until")


@dataclasses.dataclass(frozen=True, slots=True)
class JobSummary:
    """Compact job representation for listings."""

    id: str
    type: JobType
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: int | None = None
    requested_by: str | None = None
    error: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class JobTask:
    """One task inside a job run."""

    id: str
    name: str
    status: JobStatus
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: int | None = None
    output: JsonObject | None = None
    error: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class JobDetails(JobSummary):
    """Full job representation, including inputs, outputs and tasks."""

    input: JsonObject = dataclasses.field(default_factory=dict)
    output: JsonObject | None = None
    tasks: tuple[JobTask, ...] = ()
