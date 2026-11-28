"""The job use case: validate identifiers and delegate to a gateway."""

from __future__ import annotations

from lankatax_ai.backend.services.errors import ValidationError
from lankatax_ai.backend.services.jobs.deps import JobDeps
from lankatax_ai.backend.services.jobs.types import (
    JobDetails,
    JobFilter,
    JobReceipt,
    JobRequest,
    JobStatus,
    JobSummary,
)


def _require_run_id(run_id: str) -> str:
    if not run_id.strip():
        raise ValidationError("run_id must not be blank")
    return run_id


async def enqueue_job(request: JobRequest, *, deps: JobDeps) -> JobReceipt:
    """Accept a job request through the configured gateway."""
    return await deps.gateway.enqueue(request)


async def list_jobs(filters: JobFilter, *, deps: JobDeps) -> list[JobSummary]:
    """List jobs through the configured gateway."""
    return await deps.gateway.list_jobs(filters)


async def get_job(run_id: str, *, deps: JobDeps) -> JobDetails:
    """Return one job's details."""
    return await deps.gateway.get(_require_run_id(run_id))


async def get_job_status(run_id: str, *, deps: JobDeps) -> JobStatus:
    """Return one job's status."""
    return await deps.gateway.status(_require_run_id(run_id))


async def cancel_job(run_id: str, *, deps: JobDeps) -> None:
    """Request cancellation of one job."""
    await deps.gateway.cancel(_require_run_id(run_id))


async def replay_job(run_id: str, *, deps: JobDeps) -> None:
    """Replay a failed job."""
    await deps.gateway.replay(_require_run_id(run_id))
