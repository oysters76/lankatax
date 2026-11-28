"""Job logging, trace metadata and metrics."""

from __future__ import annotations

from hatchet_sdk.context.context import Context
from prometheus_client import Counter, Histogram

from lankatax_ai.backend.services.jobs.types import JobType

__all__ = [
    "job_event",
    "job_trace_metadata",
    "record_job_completed",
    "record_job_started",
]

JOBS_STARTED = Counter(
    "lankatax_job_started_total",
    "Jobs started by the Hatchet workers",
    ["job_type"],
)
JOBS_COMPLETED = Counter(
    "lankatax_job_completed_total",
    "Jobs completed by the Hatchet workers",
    ["job_type", "status"],
)
JOB_DURATION = Histogram(
    "lankatax_job_duration_seconds",
    "Job duration in seconds",
    ["job_type"],
)
JOB_TOKENS = Counter(
    "lankatax_job_tokens_total",
    "Embedding tokens consumed by jobs",
    ["job_type"],
)


def record_job_started(job_type: JobType) -> None:
    JOBS_STARTED.labels(job_type=job_type.value).inc()


def record_job_completed(
    job_type: JobType,
    *,
    status: str,
    duration_ms: int,
    tokens: int = 0,
) -> None:
    JOBS_COMPLETED.labels(job_type=job_type.value, status=status).inc()
    JOB_DURATION.labels(job_type=job_type.value).observe(duration_ms / 1000)
    if tokens:
        JOB_TOKENS.labels(job_type=job_type.value).inc(tokens)


def job_event(
    ctx: Context,
    *,
    event: str,
    job_type: JobType,
    **fields: str | float | bool | None,
) -> None:
    """Emit a structured event through the Hatchet context logger."""
    payload: dict[str, str | int | float | bool | None] = {
        "event": event,
        "job_type": job_type.value,
        "run_id": getattr(ctx, "workflow_run_id", None),
        "task_run_id": getattr(ctx, "task_run_id", None),
        "attempt": getattr(ctx, "retry_count", 0),
    }
    payload.update(fields)
    ctx.log(payload)


def job_trace_metadata(
    *,
    run_id: str,
    job_type: JobType,
    attempt: int,
    corpus_epoch: str | None = None,
) -> dict[str, str | int]:
    metadata: dict[str, str | int] = {
        "hatchet_run_id": run_id,
        "job_type": job_type.value,
        "attempt": attempt,
    }
    if corpus_epoch is not None:
        metadata["corpus_epoch"] = corpus_epoch
    return metadata
