"""Job domain types and gateway port."""

from lankatax_ai.backend.services.jobs.ports import JobGateway
from lankatax_ai.backend.services.jobs.types import (
    JobDetails,
    JobFilter,
    JobReceipt,
    JobRequest,
    JobStatus,
    JobSummary,
    JobTask,
    JobType,
    JsonObject,
    JsonValue,
)

__all__ = [
    "JobDetails",
    "JobFilter",
    "JobGateway",
    "JobReceipt",
    "JobRequest",
    "JobStatus",
    "JobSummary",
    "JobTask",
    "JobType",
    "JsonObject",
    "JsonValue",
]
