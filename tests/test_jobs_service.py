from __future__ import annotations

from datetime import UTC, datetime

import pytest

from lankatax_ai.backend.services.errors import ValidationError
from lankatax_ai.backend.services.jobs import (
    JobDeps,
    JobDetails,
    JobFilter,
    JobReceipt,
    JobRequest,
    JobStatus,
    JobSummary,
    JobType,
    cancel_job,
    enqueue_job,
    get_job,
    get_job_status,
    list_jobs,
    replay_job,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 11, 28, tzinfo=UTC)


class _FakeGateway:
    def __init__(self) -> None:
        self.request: JobRequest | None = None
        self.filters: JobFilter | None = None
        self.run_ids: list[str] = []
        self.cancelled: list[str] = []
        self.replayed: list[str] = []

    async def enqueue(self, request: JobRequest) -> JobReceipt:
        self.request = request
        return JobReceipt(run_id="run-1")

    async def list_jobs(self, filters: JobFilter) -> list[JobSummary]:
        self.filters = filters
        return [
            JobSummary(
                id="run-1",
                type=JobType.pipeline_refresh,
                status=JobStatus.queued,
                created_at=NOW,
            )
        ]

    async def get(self, run_id: str) -> JobDetails:
        self.run_ids.append(run_id)
        return JobDetails(
            id=run_id,
            type=JobType.pipeline_refresh,
            status=JobStatus.running,
            created_at=NOW,
        )

    async def status(self, run_id: str) -> JobStatus:
        self.run_ids.append(run_id)
        return JobStatus.running

    async def cancel(self, run_id: str) -> None:
        self.cancelled.append(run_id)

    async def replay(self, run_id: str) -> None:
        self.replayed.append(run_id)


def _deps() -> tuple[JobDeps, _FakeGateway]:
    gateway = _FakeGateway()
    return JobDeps(gateway=gateway), gateway


async def test_enqueue_delegates_and_returns_receipt():
    deps, gateway = _deps()
    request = JobRequest(type=JobType.scrape_registry, params={"full": True})

    receipt = await enqueue_job(request, deps=deps)

    assert receipt == JobReceipt(run_id="run-1")
    assert gateway.request is request


async def test_list_delegates_with_filters():
    deps, gateway = _deps()
    filters = JobFilter(limit=10)

    summaries = await list_jobs(filters, deps=deps)

    assert gateway.filters is filters
    assert [summary.id for summary in summaries] == ["run-1"]


async def test_get_and_status_delegate():
    deps, gateway = _deps()

    details = await get_job("run-1", deps=deps)
    status = await get_job_status("run-2", deps=deps)

    assert details.id == "run-1"
    assert status is JobStatus.running
    assert gateway.run_ids == ["run-1", "run-2"]


async def test_cancel_and_replay_delegate():
    deps, gateway = _deps()

    await cancel_job("run-1", deps=deps)
    await replay_job("run-2", deps=deps)

    assert gateway.cancelled == ["run-1"]
    assert gateway.replayed == ["run-2"]


@pytest.mark.parametrize(
    "operation",
    [get_job, get_job_status, cancel_job, replay_job],
)
async def test_blank_run_id_is_rejected(operation):
    deps, _gateway = _deps()

    with pytest.raises(ValidationError, match="run_id"):
        await operation("   ", deps=deps)
