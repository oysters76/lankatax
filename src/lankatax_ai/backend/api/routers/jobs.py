"""Job administration endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from lankatax_ai.backend.api.deps import ApiDepsDep
from lankatax_ai.backend.api.schemas import (
    JobActionSchema,
    JobDetailsSchema,
    JobListSchema,
    JobReceiptSchema,
    JobRequestSchema,
    JobSummarySchema,
    JobTaskSchema,
    Link,
)
from lankatax_ai.backend.api.security import require_access
from lankatax_ai.backend.api.utils import link
from lankatax_ai.backend.config import settings
from lankatax_ai.backend.services.jobs import (
    JobDetails,
    JobFilter,
    JobRequest,
    JobStatus,
    JobSummary,
    JobTask,
    JobType,
    cancel_job,
    enqueue_job,
    get_job,
    list_jobs,
    replay_job,
)

router = APIRouter(tags=["jobs"])

_PROTECTED = [Depends(require_access)]


def _summary_links(request: Request, summary: JobSummary) -> dict[str, Link]:
    links = {
        "self": link(request, "job_detail", run_id=summary.id),
        "api_root": link(request, "api_root"),
    }
    if summary.status in (JobStatus.queued, JobStatus.running):
        links["cancel"] = link(
            request,
            "job_cancel",
            run_id=summary.id,
            method="POST",
            title="Cancel job",
        )
    if summary.status in (JobStatus.failed, JobStatus.cancelled):
        links["replay"] = link(
            request,
            "job_replay",
            run_id=summary.id,
            method="POST",
            title="Replay job",
        )
    return links


def _summary_schema(request: Request, summary: JobSummary) -> JobSummarySchema:
    return JobSummarySchema(
        id=summary.id,
        type=summary.type,
        status=summary.status,
        created_at=summary.created_at,
        started_at=summary.started_at,
        finished_at=summary.finished_at,
        duration_ms=summary.duration_ms,
        requested_by=summary.requested_by,
        error=summary.error,
        links=_summary_links(request, summary),
    )


def _task_schema(task: JobTask) -> JobTaskSchema:
    return JobTaskSchema(
        id=task.id,
        name=task.name,
        status=task.status,
        started_at=task.started_at,
        finished_at=task.finished_at,
        duration_ms=task.duration_ms,
        output=task.output,
        error=task.error,
    )


def _details_schema(request: Request, details: JobDetails) -> JobDetailsSchema:
    links = _summary_links(request, details)
    if settings.hatchet_dashboard_url:
        links["dashboard"] = Link(
            href=f"{settings.hatchet_dashboard_url.rstrip('/')}/runs/{details.id}",
            title="Hatchet run",
        )
    return JobDetailsSchema(
        id=details.id,
        type=details.type,
        status=details.status,
        created_at=details.created_at,
        started_at=details.started_at,
        finished_at=details.finished_at,
        duration_ms=details.duration_ms,
        requested_by=details.requested_by,
        error=details.error,
        input=details.input,
        output=details.output,
        tasks=[_task_schema(task) for task in details.tasks],
        links=links,
    )


@router.post(
    "/api/v1/jobs",
    response_model=JobReceiptSchema,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue a job",
    dependencies=_PROTECTED,
)
async def jobs(request: Request, payload: JobRequestSchema, deps: ApiDepsDep) -> JSONResponse:
    receipt = await enqueue_job(
        JobRequest(
            type=payload.type,
            params=payload.params,
            idempotency_key=payload.idempotency_key,
        ),
        deps=deps.jobs,
    )
    detail = link(request, "job_detail", run_id=receipt.run_id)
    body = JobReceiptSchema(
        run_id=receipt.run_id,
        reused=receipt.reused,
        links={"self": detail, "api_root": link(request, "api_root")},
    )
    headers = {"Location": detail.href}
    if receipt.reused:
        headers["Idempotent-Replayed"] = "true"
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content=jsonable_encoder(body),
        headers=headers,
    )


@router.get(
    "/api/v1/jobs",
    response_model=JobListSchema,
    summary="List jobs",
    dependencies=_PROTECTED,
)
async def jobs_list(
    request: Request,
    deps: ApiDepsDep,
    statuses: Annotated[list[JobStatus] | None, Query(alias="status")] = None,
    types: Annotated[list[JobType] | None, Query(alias="type")] = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> JobListSchema:
    filters = JobFilter(
        statuses=tuple(statuses or ()),
        types=tuple(types or ()),
        since=since,
        until=until,
        limit=limit,
        offset=offset,
    )
    summaries = await list_jobs(filters, deps=deps.jobs)
    next_link = (
        Link(
            href=str(request.url.include_query_params(offset=offset + limit)),
            title="Next page",
        )
        if len(summaries) == limit
        else None
    )
    return JobListSchema(
        items=[_summary_schema(request, summary) for summary in summaries],
        next=next_link,
        links={"self": link(request, "jobs"), "api_root": link(request, "api_root")},
    )


@router.get(
    "/api/v1/jobs/{run_id}",
    response_model=JobDetailsSchema,
    summary="Get a job",
    dependencies=_PROTECTED,
)
async def job_detail(request: Request, run_id: str, deps: ApiDepsDep) -> JobDetailsSchema:
    details = await get_job(run_id, deps=deps.jobs)
    return _details_schema(request, details)


@router.post(
    "/api/v1/jobs/{run_id}/cancel",
    response_model=JobActionSchema,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Cancel a job",
    dependencies=_PROTECTED,
)
async def job_cancel(request: Request, run_id: str, deps: ApiDepsDep) -> JSONResponse:
    await cancel_job(run_id, deps=deps.jobs)
    body = JobActionSchema(
        run_id=run_id,
        action="cancel",
        links={
            "self": link(request, "job_cancel", run_id=run_id, method="POST"),
            "job": link(request, "job_detail", run_id=run_id),
            "api_root": link(request, "api_root"),
        },
    )
    return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=jsonable_encoder(body))


@router.post(
    "/api/v1/jobs/{run_id}/replay",
    response_model=JobActionSchema,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Replay a job",
    dependencies=_PROTECTED,
)
async def job_replay(request: Request, run_id: str, deps: ApiDepsDep) -> JSONResponse:
    await replay_job(run_id, deps=deps.jobs)
    body = JobActionSchema(
        run_id=run_id,
        action="replay",
        links={
            "self": link(request, "job_replay", run_id=run_id, method="POST"),
            "job": link(request, "job_detail", run_id=run_id),
            "api_root": link(request, "api_root"),
        },
    )
    return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=jsonable_encoder(body))
