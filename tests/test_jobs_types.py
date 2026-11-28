from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from lankatax_ai.backend.services.errors import ValidationError
from lankatax_ai.backend.services.jobs import (
    JobDetails,
    JobFilter,
    JobGateway,
    JobRequest,
    JobStatus,
    JobSummary,
    JobTask,
    JobType,
)
from lankatax_ai.backend.services.jobs.limits import DEFAULT_LIMIT, MAX_LIMIT, MIN_LIMIT

pytestmark = pytest.mark.unit

NOW = datetime(2026, 11, 28, tzinfo=UTC)


def test_job_type_values_are_wire_values():
    assert {job_type.value for job_type in JobType} == {
        "pipeline.refresh",
        "scrape.registry",
        "ingest.pending",
        "ingest.document",
        "cache.expire",
    }


def test_job_status_values_are_wire_values():
    assert {status.value for status in JobStatus} == {
        "queued",
        "running",
        "succeeded",
        "succeeded_with_errors",
        "failed",
        "cancelled",
    }


def test_job_request_defaults():
    request = JobRequest(type=JobType.pipeline_refresh)

    assert request.params == {}
    assert request.idempotency_key is None
    assert request.requested_by is None


def test_job_request_rejects_blank_idempotency_key():
    with pytest.raises(ValidationError, match="idempotency_key"):
        JobRequest(type=JobType.pipeline_refresh, idempotency_key="   ")


def test_job_filter_defaults():
    filters = JobFilter()

    assert filters.statuses == ()
    assert filters.types == ()
    assert filters.since is None
    assert filters.until is None
    assert filters.limit == DEFAULT_LIMIT
    assert filters.offset == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"limit": MIN_LIMIT - 1}, "Limit must be 1-100"),
        ({"limit": MAX_LIMIT + 1}, "Limit must be 1-100"),
        ({"offset": -1}, "Offset must not be negative"),
        (
            {"since": NOW, "until": NOW - timedelta(seconds=1)},
            "since must not be after until",
        ),
    ],
)
def test_job_filter_validation(kwargs, message):
    with pytest.raises(ValidationError, match=message):
        JobFilter(**kwargs)


def test_job_filter_accepts_boundaries():
    filters = JobFilter(limit=1, offset=0)
    assert filters.limit == 1

    filters = JobFilter(limit=100)
    assert filters.limit == 100


def test_job_summary_and_details_are_immutable_slots():
    summary = JobSummary(
        id="run-1",
        type=JobType.scrape_registry,
        status=JobStatus.running,
        created_at=NOW,
    )
    task = JobTask(
        id="task-1",
        name="scrape-registry",
        status=JobStatus.succeeded,
        output={"saved": 3},
    )
    details = JobDetails(
        **dataclasses.asdict(summary),
        input={"full": True},
        output={"saved": 3},
        tasks=(task,),
    )

    assert not hasattr(summary, "__dict__")
    assert not hasattr(details, "__dict__")
    assert details.tasks == (task,)
    assert details.input == {"full": True}

    with pytest.raises(dataclasses.FrozenInstanceError):
        summary.status = JobStatus.failed  # type: ignore[misc]


class _FakeGateway:
    async def enqueue(self, request: JobRequest) -> str:
        return "run-1"

    async def list(self, filters: JobFilter) -> list[JobSummary]:
        return []

    async def get(self, run_id: str) -> JobDetails:
        raise AssertionError("not used")

    async def status(self, run_id: str) -> JobStatus:
        return JobStatus.queued

    async def cancel(self, run_id: str) -> None:
        return None

    async def replay(self, run_id: str) -> None:
        return None


async def test_gateway_port_is_implementable_without_an_sdk():
    gateway: JobGateway = _FakeGateway()

    assert await gateway.enqueue(JobRequest(type=JobType.pipeline_refresh)) == "run-1"
    assert await gateway.list(JobFilter()) == []
    assert await gateway.status("run-1") is JobStatus.queued
