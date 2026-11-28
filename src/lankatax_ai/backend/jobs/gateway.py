"""Hatchet implementation of the job gateway port."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import cast

from hatchet_sdk import Hatchet, IdempotencyCollisionError
from hatchet_sdk.clients.rest.exceptions import (
    ApiException,
    BadRequestException,
    NotFoundException,
    UnprocessableEntityException,
)
from hatchet_sdk.clients.rest.models.v1_task_status import V1TaskStatus
from hatchet_sdk.clients.rest.models.v1_task_summary import V1TaskSummary
from hatchet_sdk.exceptions import DedupeViolationError
from pydantic import ValidationError as PydanticValidationError

from lankatax_ai.backend.jobs.contracts import serialize_input
from lankatax_ai.backend.services.errors import (
    NotFoundError,
    ServiceError,
    UpstreamError,
    ValidationError,
)
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
)

__all__ = ["HatchetJobGateway"]

logger = logging.getLogger(__name__)

_JOB_TYPES_BY_VALUE = {job_type.value: job_type for job_type in JobType}


class HatchetJobGateway(JobGateway):
    """Translate the domain job vocabulary to Hatchet runs."""

    def __init__(self, client: Hatchet) -> None:
        self._client = client

    async def enqueue(self, request: JobRequest) -> JobReceipt:
        try:
            payload = serialize_input(request.type, request.params)
        except PydanticValidationError as error:
            raise ValidationError(f"Invalid {request.type.value} parameters: {error}") from error

        idempotency_key = request.idempotency_key or self._payload_key(payload)
        metadata: dict[str, str] = {"job_type": request.type.value}
        if request.requested_by:
            metadata["requested_by"] = request.requested_by
        if idempotency_key:
            metadata["idempotency_key"] = idempotency_key

        try:
            details = await self._call(
                lambda: self._client.runs.aio_create(
                    workflow_name=request.type.value,
                    input=payload,
                    additional_metadata=metadata,
                )
            )
        except IdempotencyCollisionError as error:
            return JobReceipt(run_id=error.existing_run_external_id, reused=True)
        except UpstreamError:
            # Lite reports some collisions as a plain 500, so fall back to the
            # active run recorded under the same idempotency key.
            existing = await self._resolve_active_run(request.type, idempotency_key)
            if existing is None:
                raise
            return JobReceipt(run_id=existing, reused=True)
        except DedupeViolationError as error:
            raise UpstreamError(
                "Hatchet rejected a duplicate run without a run reference"
            ) from error
        except (TypeError, ValueError) as error:
            raise ValidationError("Job parameters are not JSON-serializable") from error

        return JobReceipt(run_id=details.run.metadata.id)

    async def _resolve_active_run(
        self,
        job_type: JobType,
        idempotency_key: str | None,
    ) -> str | None:
        """Find the queued or running run holding an idempotency key."""
        if not idempotency_key:
            return None
        try:
            by_type, _ = await self._workflow_index()
            workflow_id = by_type.get(job_type)
            if workflow_id is None:
                return None
            response = await self._call(
                lambda: self._client.runs.aio_list(
                    workflow_ids=[workflow_id],
                    statuses=[V1TaskStatus.QUEUED, V1TaskStatus.RUNNING],
                    only_tasks=False,
                    include_payloads=True,
                    limit=50,
                )
            )
        except ServiceError:
            logger.warning("Could not resolve the idempotent run", exc_info=True)
            return None

        for row in response.rows or []:
            if self._metadata_str(row.additional_metadata, "idempotency_key") == idempotency_key:
                return row.workflow_run_external_id or row.metadata.id
        return None

    @staticmethod
    def _payload_key(payload: Mapping[str, object]) -> str | None:
        value = payload.get("idempotency_key")
        return value if isinstance(value, str) and value else None

    async def list_jobs(self, filters: JobFilter) -> list[JobSummary]:
        by_type, by_id = await self._workflow_index()
        if filters.types:
            workflow_ids = [by_type[job_type] for job_type in filters.types if job_type in by_type]
        else:
            workflow_ids = list(by_type.values())

        if not workflow_ids:
            return []

        response = await self._call(
            lambda: self._client.runs.aio_list(
                since=filters.since,
                until=filters.until,
                offset=filters.offset,
                limit=filters.limit,
                statuses=self._sdk_statuses(filters.statuses),
                workflow_ids=workflow_ids,
                only_tasks=False,
                include_payloads=True,
            )
        )

        summaries: list[JobSummary] = []
        seen: set[str] = set()
        for row in response.rows or []:
            try:
                run_id = row.workflow_run_external_id or row.metadata.id
                if run_id in seen:
                    continue
                seen.add(run_id)
                job_type = self._job_type(
                    row.additional_metadata,
                    row.workflow_name,
                    by_id.get(row.workflow_id),
                )
                if job_type is None:
                    logger.warning(
                        "Skipping unmapped workflow run %s (%s)",
                        run_id,
                        getattr(row, "workflow_name", None),
                    )
                    continue

                summary = self._summary_from_row(row, job_type)
            except AttributeError:
                logger.exception("Unexpected Hatchet row shape while listing runs")
                raise

            if filters.statuses and summary.status not in filters.statuses:
                continue
            summaries.append(summary)
        return summaries

    async def get(self, run_id: str) -> JobDetails:
        details = await self._call(lambda: self._client.runs.aio_get(run_id))

        try:
            _by_type, by_id = await self._workflow_index()
            run = details.run
            workflow_name = by_id.get(run.workflow_id)
            job_type = self._job_type(
                run.additional_metadata,
                workflow_name,
            )
            if job_type is None:
                raise UpstreamError(
                    f"Unmapped workflow {workflow_name or run.workflow_id!r} for run {run_id}"
                )

            tasks = tuple(self._task_from_row(task) for task in details.tasks)
            output = self._optional_json_object(run.output, f"run {run_id} output")
            return JobDetails(
                id=run.metadata.id,
                type=job_type,
                status=self._status(run.status, run.output),
                created_at=run.metadata.created_at,
                started_at=run.started_at,
                finished_at=run.finished_at,
                duration_ms=run.duration,
                requested_by=self._metadata_str(run.additional_metadata, "requested_by"),
                error=run.error_message,
                input=self._input_object(run.input, f"run {run_id}"),
                output=output or None,
                tasks=tasks,
            )
        except AttributeError:
            logger.exception("Unexpected Hatchet run shape for %s", run_id)
            raise

    async def status(self, run_id: str) -> JobStatus:
        status = await self._call(lambda: self._client.runs.aio_get_status(run_id))
        return self._status(status, None)

    async def cancel(self, run_id: str) -> None:
        await self._call(lambda: self._client.runs.aio_cancel(run_id))

    async def replay(self, run_id: str) -> None:
        await self._call(lambda: self._client.runs.aio_replay(run_id))

    async def _call[T](self, operation: Callable[[], Awaitable[T]]) -> T:
        try:
            return await operation()
        except ApiException as error:
            raise self._service_error(error) from error
        except IdempotencyCollisionError:
            raise
        except DedupeViolationError:
            raise
        except Exception:
            logger.exception("Unexpected Hatchet operation failure")
            raise

    @staticmethod
    def _service_error(error: ApiException) -> ServiceError:
        if isinstance(error, (BadRequestException, UnprocessableEntityException)):
            return ValidationError("Hatchet rejected the job request")
        if isinstance(error, NotFoundException):
            return NotFoundError("Job run not found")
        if error.status in (401, 403):
            return UpstreamError("Hatchet authorization failed")
        return UpstreamError(f"Hatchet request failed with status {error.status or 'unknown'}")

    async def _workflow_index(self) -> tuple[dict[JobType, str], dict[str, str]]:
        response = await self._call(lambda: self._client.workflows.aio_list(limit=100))
        by_type: dict[JobType, str] = {}
        by_id: dict[str, str] = {}
        for workflow in response.rows or []:
            by_id[workflow.metadata.id] = workflow.name
            job_type = _JOB_TYPES_BY_VALUE.get(workflow.name)
            if job_type is None:
                logger.debug("Ignoring non-job workflow %r", workflow.name)
                continue
            by_type[job_type] = workflow.metadata.id
        return by_type, by_id

    def _summary_from_row(self, row: V1TaskSummary, job_type: JobType) -> JobSummary:
        return JobSummary(
            id=row.workflow_run_external_id or row.metadata.id,
            type=job_type,
            status=self._status(row.status, row.output),
            created_at=row.created_at or row.metadata.created_at,
            started_at=row.started_at,
            finished_at=row.finished_at,
            duration_ms=row.duration,
            requested_by=self._metadata_str(row.additional_metadata, "requested_by"),
            error=row.error_message,
        )

    def _task_from_row(self, task: V1TaskSummary) -> JobTask:
        return JobTask(
            id=task.metadata.id,
            name=str(task.display_name or task.task_id or task.step_id or task.metadata.id),
            status=self._status(task.status, task.output),
            started_at=task.started_at,
            finished_at=task.finished_at,
            duration_ms=task.duration,
            output=self._optional_json_object(
                task.output,
                f"task {task.metadata.id} output",
            )
            or None,
            error=task.error_message,
        )

    @staticmethod
    def _status(status: V1TaskStatus, output: Mapping[str, object] | None) -> JobStatus:
        if status is V1TaskStatus.QUEUED:
            return JobStatus.queued
        if status is V1TaskStatus.RUNNING:
            return JobStatus.running
        if status is V1TaskStatus.CANCELLED:
            return JobStatus.cancelled
        if status is V1TaskStatus.FAILED:
            return JobStatus.failed
        if status is V1TaskStatus.COMPLETED:
            if HatchetJobGateway._is_partial(output):
                return JobStatus.succeeded_with_errors
            return JobStatus.succeeded
        raise UpstreamError(f"Unknown Hatchet status: {status}")

    @staticmethod
    def _sdk_statuses(statuses: tuple[JobStatus, ...]) -> list[V1TaskStatus] | None:
        if not statuses:
            return None
        mapped: list[V1TaskStatus] = []
        for status in statuses:
            if status is JobStatus.queued:
                mapped.append(V1TaskStatus.QUEUED)
            elif status is JobStatus.running:
                mapped.append(V1TaskStatus.RUNNING)
            elif status is JobStatus.cancelled:
                mapped.append(V1TaskStatus.CANCELLED)
            elif status is JobStatus.failed:
                mapped.append(V1TaskStatus.FAILED)
            else:
                mapped.append(V1TaskStatus.COMPLETED)
        return mapped

    @staticmethod
    def _job_type(
        metadata: Mapping[str, object] | None,
        *workflow_names: str | None,
    ) -> JobType | None:
        value = HatchetJobGateway._metadata_str(metadata, "job_type")
        if value is not None:
            try:
                return JobType(value)
            except ValueError:
                logger.warning("Ignoring unknown job_type metadata %r", value)

        for workflow_name in workflow_names:
            if workflow_name is None:
                continue
            try:
                return JobType(workflow_name)
            except ValueError:
                logger.debug("Workflow %r is not a known job type", workflow_name)
                continue
        return None

    @staticmethod
    def _metadata_str(metadata: Mapping[str, object] | None, key: str) -> str | None:
        if metadata is None:
            return None
        value = metadata.get(key)
        if value is None or isinstance(value, str):
            return value
        logger.warning("Ignoring non-string metadata %s=%r", key, value)
        return None

    @staticmethod
    def _input_object(value: object | None, context: str) -> JsonObject:
        if isinstance(value, Mapping):
            return cast(JsonObject, value)
        raise UpstreamError(f"Hatchet returned non-object input for {context}")

    @staticmethod
    def _optional_json_object(value: object | None, context: str) -> JsonObject | None:
        if value is None:
            return None
        if isinstance(value, Mapping):
            return cast(JsonObject, value)
        logger.warning(
            "Ignoring non-object JSON payload for %s: %s",
            context,
            type(value).__name__,
        )
        return None

    @staticmethod
    def _is_partial(value: object | None) -> bool:
        if isinstance(value, Mapping):
            if value.get("partial") is True or value.get("budget_exhausted") is True:
                return True
            return any(HatchetJobGateway._is_partial(item) for item in value.values())
        if isinstance(value, list):
            return any(HatchetJobGateway._is_partial(item) for item in value)
        return False
