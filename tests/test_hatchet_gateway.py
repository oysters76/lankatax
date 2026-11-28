from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from hatchet_sdk import IdempotencyCollisionError
from hatchet_sdk.clients.rest.exceptions import (
    ApiException,
    BadRequestException,
    NotFoundException,
)
from hatchet_sdk.clients.rest.models.v1_task_status import V1TaskStatus
from hatchet_sdk.exceptions import DedupeViolationError

from lankatax_ai.backend.jobs.gateway import HatchetJobGateway
from lankatax_ai.backend.services.errors import (
    NotFoundError,
    UpstreamError,
    ValidationError,
)
from lankatax_ai.backend.services.jobs import (
    JobFilter,
    JobRequest,
    JobStatus,
    JobType,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 11, 28, tzinfo=UTC)


def _workflow(name: str, workflow_id: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, metadata=SimpleNamespace(id=workflow_id))


def _row(
    *,
    run_id: str = "run-1",
    workflow_id: str = "wf-refresh",
    name: str = "pipeline.refresh",
    status: V1TaskStatus = V1TaskStatus.COMPLETED,
    output: dict[str, Any] | None = None,
    metadata: dict[str, str] | None = None,
    workflow_run_external_id: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        metadata=SimpleNamespace(id=run_id, created_at=NOW),
        workflow_run_external_id=workflow_run_external_id,
        workflow_id=workflow_id,
        workflow_name=name,
        status=status,
        created_at=NOW,
        started_at=NOW,
        finished_at=NOW,
        duration=1000,
        additional_metadata=metadata
        if metadata is not None
        else {"job_type": name, "requested_by": "admin"},
        output=output or {},
        error_message=None,
        display_name="scrape-registry",
        task_id="task-id",
        step_id="step-id",
    )


class _FakeRuns:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.list_kwargs: dict[str, Any] = {}
        self.rows: list[SimpleNamespace] = []
        self.details: SimpleNamespace | None = None
        self.status_value = V1TaskStatus.QUEUED
        self.cancelled: list[str] = []
        self.replayed: list[str] = []
        self.collision: str | None = None
        self.create_error: Exception | None = None
        self.get_error: Exception | None = None

    async def aio_create(self, **kwargs: Any) -> SimpleNamespace:
        self.created.append(kwargs)
        if self.create_error is not None:
            raise self.create_error
        if self.collision is not None:
            raise IdempotencyCollisionError(self.collision)
        return SimpleNamespace(run=SimpleNamespace(metadata=SimpleNamespace(id="run-1")))

    async def aio_list(self, **kwargs: Any) -> SimpleNamespace:
        self.list_kwargs = kwargs
        return SimpleNamespace(rows=self.rows)

    async def aio_get(self, run_id: str) -> SimpleNamespace:
        if self.get_error is not None:
            raise self.get_error
        assert self.details is not None
        return self.details

    async def aio_get_status(self, run_id: str) -> V1TaskStatus:
        return self.status_value

    async def aio_cancel(self, run_id: str) -> None:
        self.cancelled.append(run_id)

    async def aio_replay(self, run_id: str) -> None:
        self.replayed.append(run_id)


class _FakeWorkflows:
    def __init__(self) -> None:
        self.rows = [
            _workflow("pipeline.refresh", "wf-refresh"),
            _workflow("scrape.registry", "wf-scrape"),
        ]
        self.error: Exception | None = None

    async def aio_list(self, limit: int | None = None) -> SimpleNamespace:
        if self.error is not None:
            raise self.error
        return SimpleNamespace(rows=self.rows)


class _FakeClient:
    def __init__(self) -> None:
        self.runs = _FakeRuns()
        self.workflows = _FakeWorkflows()


def _gateway() -> tuple[HatchetJobGateway, _FakeClient]:
    client = _FakeClient()
    return HatchetJobGateway(client), client  # type: ignore[arg-type]


async def test_enqueue_maps_request_to_hatchet():
    gateway, client = _gateway()

    receipt = await gateway.enqueue(
        JobRequest(
            type=JobType.pipeline_refresh,
            params={"full": True},
            idempotency_key="key-1",
            requested_by="admin",
        )
    )

    assert receipt.run_id == "run-1"
    assert receipt.reused is False
    assert client.runs.created == [
        {
            "workflow_name": "pipeline.refresh",
            "input": {"full": True},
            "additional_metadata": {
                "job_type": "pipeline.refresh",
                "requested_by": "admin",
                "idempotency_key": "key-1",
            },
        }
    ]


async def test_enqueue_returns_existing_run_on_collision():
    gateway, client = _gateway()
    client.runs.collision = "existing-run"

    receipt = await gateway.enqueue(JobRequest(type=JobType.pipeline_refresh))

    assert receipt.run_id == "existing-run"
    assert receipt.reused is True


async def test_enqueue_maps_bad_request_to_validation_error():
    gateway, client = _gateway()
    client.runs.create_error = BadRequestException(status=400)

    with pytest.raises(ValidationError, match="rejected"):
        await gateway.enqueue(JobRequest(type=JobType.pipeline_refresh))


async def test_enqueue_maps_serialization_error_to_validation_error():
    gateway, client = _gateway()
    client.runs.create_error = TypeError("not json")

    with pytest.raises(ValidationError, match="JSON-serializable"):
        await gateway.enqueue(JobRequest(type=JobType.pipeline_refresh))


async def test_enqueue_maps_dedupe_error_to_upstream_error():
    gateway, client = _gateway()
    client.runs.create_error = DedupeViolationError()

    with pytest.raises(UpstreamError, match="duplicate run"):
        await gateway.enqueue(JobRequest(type=JobType.pipeline_refresh))


async def test_list_maps_workflow_lookup_failure_to_upstream_error():
    gateway, client = _gateway()
    client.workflows.error = ApiException(status=503)

    with pytest.raises(UpstreamError, match="status 503"):
        await gateway.list_jobs(JobFilter())


async def test_get_maps_not_found():
    gateway, client = _gateway()
    client.runs.get_error = NotFoundException(status=404)

    with pytest.raises(NotFoundError, match="Job run not found"):
        await gateway.get("missing")


async def test_get_rejects_non_object_input():
    gateway, client = _gateway()
    client.runs.details = SimpleNamespace(
        run=SimpleNamespace(
            metadata=SimpleNamespace(id="run-1", created_at=NOW),
            status=V1TaskStatus.COMPLETED,
            started_at=NOW,
            finished_at=NOW,
            duration=1,
            additional_metadata={"job_type": "pipeline.refresh"},
            workflow_id="wf-refresh",
            input=["not", "an", "object"],
            output={},
            error_message=None,
        ),
        tasks=[],
    )

    with pytest.raises(UpstreamError, match="non-object input"):
        await gateway.get("run-1")


def test_unknown_status_is_not_treated_as_success():
    with pytest.raises(UpstreamError, match="Unknown Hatchet status"):
        HatchetJobGateway._status("BOGUS", None)  # type: ignore[arg-type]


async def test_list_logs_skipped_unmapped_workflow(caplog):
    gateway, client = _gateway()
    client.runs.rows = [
        _row(
            run_id="run-unknown",
            workflow_id="wf-unknown",
            name="other.workflow",
            metadata={},
        )
    ]

    summaries = await gateway.list_jobs(JobFilter())

    assert summaries == []
    assert "Skipping unmapped workflow run run-unknown" in caplog.text


async def test_get_maps_unmapped_workflow_to_upstream_error():
    gateway, client = _gateway()
    client.runs.details = SimpleNamespace(
        run=SimpleNamespace(
            metadata=SimpleNamespace(id="run-1", created_at=NOW),
            status=V1TaskStatus.COMPLETED,
            started_at=NOW,
            finished_at=NOW,
            duration=1,
            additional_metadata={},
            workflow_id="wf-unknown",
            input={},
            output={},
            error_message=None,
        ),
        tasks=[],
    )

    with pytest.raises(UpstreamError, match="Unmapped workflow"):
        await gateway.get("run-1")


async def test_list_maps_and_deduplicates_workflow_runs():
    gateway, client = _gateway()
    client.runs.rows = [
        _row(output={"partial": True}),
        _row(run_id="run-1", workflow_run_external_id="run-1"),
        _row(
            run_id="run-2",
            workflow_id="wf-scrape",
            name="scrape.registry",
            status=V1TaskStatus.FAILED,
        ),
    ]

    summaries = await gateway.list_jobs(JobFilter())

    assert [summary.id for summary in summaries] == ["run-1", "run-2"]
    assert summaries[0].type is JobType.pipeline_refresh
    assert summaries[0].status is JobStatus.succeeded_with_errors
    assert summaries[1].status is JobStatus.failed
    assert client.runs.list_kwargs["workflow_ids"] == ["wf-refresh", "wf-scrape"]


async def test_list_filters_types_to_known_workflows():
    gateway, client = _gateway()

    summaries = await gateway.list_jobs(JobFilter(types=(JobType.scrape_registry,)))

    assert summaries == []
    assert client.runs.list_kwargs["workflow_ids"] == ["wf-scrape"]


async def test_list_filters_partial_status_exactly():
    gateway, client = _gateway()
    client.runs.rows = [
        _row(run_id="partial", output={"partial": True}),
        _row(run_id="clean", output={}),
    ]

    summaries = await gateway.list_jobs(JobFilter(statuses=(JobStatus.succeeded_with_errors,)))

    assert [summary.id for summary in summaries] == ["partial"]


async def test_get_maps_run_and_tasks():
    gateway, client = _gateway()
    client.runs.details = SimpleNamespace(
        run=SimpleNamespace(
            metadata=SimpleNamespace(id="run-1", created_at=NOW),
            status=V1TaskStatus.COMPLETED,
            started_at=NOW,
            finished_at=NOW,
            duration=2500,
            additional_metadata={"job_type": "pipeline.refresh", "requested_by": "admin"},
            workflow_id="wf-refresh",
            input={"full": True},
            output={"partial": True},
            error_message=None,
        ),
        tasks=[_row(output={"saved": 3})],
    )

    details = await gateway.get("run-1")

    assert details.id == "run-1"
    assert details.type is JobType.pipeline_refresh
    assert details.status is JobStatus.succeeded_with_errors
    assert details.input == {"full": True}
    assert details.output == {"partial": True}
    assert details.requested_by == "admin"
    assert len(details.tasks) == 1
    assert details.tasks[0].name == "scrape-registry"


@pytest.mark.parametrize(
    ("sdk_status", "domain_status"),
    [
        (V1TaskStatus.QUEUED, JobStatus.queued),
        (V1TaskStatus.RUNNING, JobStatus.running),
        (V1TaskStatus.COMPLETED, JobStatus.succeeded),
        (V1TaskStatus.CANCELLED, JobStatus.cancelled),
        (V1TaskStatus.FAILED, JobStatus.failed),
    ],
)
async def test_status_mapping(sdk_status, domain_status):
    gateway, client = _gateway()
    client.runs.status_value = sdk_status

    assert await gateway.status("run-1") is domain_status


async def test_cancel_and_replay_delegate_to_hatchet():
    gateway, client = _gateway()

    await gateway.cancel("run-1")
    await gateway.replay("run-1")

    assert client.runs.cancelled == ["run-1"]
    assert client.runs.replayed == ["run-1"]
