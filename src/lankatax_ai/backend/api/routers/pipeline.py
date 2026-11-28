"""Pipeline refresh endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from lankatax_ai.backend.api.deps import ApiDepsDep
from lankatax_ai.backend.api.schemas import JobReceiptSchema, PipelineRefreshRequest
from lankatax_ai.backend.api.security import require_access
from lankatax_ai.backend.api.utils import job_receipt_response
from lankatax_ai.backend.services.jobs import JobRequest, JobType, enqueue_job

router = APIRouter(tags=["pipeline"])


@router.post(
    "/api/v1/pipeline/refresh",
    response_model=JobReceiptSchema,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Refresh the corpus",
    dependencies=[Depends(require_access)],
)
async def pipeline_refresh(
    request: Request,
    payload: PipelineRefreshRequest,
    deps: ApiDepsDep,
) -> JSONResponse:
    params = payload.model_dump(exclude={"idempotency_key"}, exclude_none=True)
    if payload.idempotency_key:
        params["idempotency_key"] = payload.idempotency_key
    receipt = await enqueue_job(
        JobRequest(
            type=JobType.pipeline_refresh,
            params=params,
            idempotency_key=payload.idempotency_key,
        ),
        deps=deps.jobs,
    )
    return job_receipt_response(request, receipt)
