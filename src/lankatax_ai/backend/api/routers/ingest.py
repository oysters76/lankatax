"""Ingest job and status endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from lankatax_ai.backend.api.deps import ApiDepsDep
from lankatax_ai.backend.api.schemas import (
    IngestRequest,
    IngestStatusSchema,
    JobReceiptSchema,
)
from lankatax_ai.backend.api.security import require_access
from lankatax_ai.backend.api.utils import job_receipt_response, link
from lankatax_ai.backend.services.ingest import get_ingestion_status
from lankatax_ai.backend.services.jobs import JobRequest, JobType, enqueue_job

router = APIRouter(tags=["ingest"])

_PROTECTED = [Depends(require_access)]


@router.post(
    "/api/v1/ingest",
    response_model=JobReceiptSchema,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest documents",
    dependencies=_PROTECTED,
)
async def ingest(request: Request, payload: IngestRequest, deps: ApiDepsDep) -> JSONResponse:
    idempotency_key = payload.idempotency_key

    if payload.document_id is not None:
        job_type = JobType.ingest_document
        params = {
            "document_id": str(payload.document_id),
            "max_tokens": payload.max_tokens,
            "batch_size": payload.batch_size,
        }
        idempotency_key = None
    else:
        job_type = JobType.ingest_pending
        params = {
            "idempotency_key": idempotency_key,
            "target": payload.target,
            "max_documents": payload.max_documents,
            "max_tokens": payload.max_tokens,
            "batch_size": payload.batch_size,
        }

    receipt = await enqueue_job(
        JobRequest(
            type=job_type,
            params={key: value for key, value in params.items() if value is not None},
            idempotency_key=idempotency_key,
        ),
        deps=deps.jobs,
    )
    return job_receipt_response(request, receipt)


@router.get(
    "/api/v1/ingest/status",
    response_model=IngestStatusSchema,
    summary="Ingestion status",
    dependencies=_PROTECTED,
)
async def ingest_status(request: Request, deps: ApiDepsDep) -> IngestStatusSchema:
    result = await get_ingestion_status(
        session_factory=deps.session_factory,
        deps=deps.ingest,
    )
    return IngestStatusSchema(
        total_documents=result.total_documents,
        pending=result.pending,
        ingested=result.ingested,
        links={
            "self": link(request, "ingest_status"),
            "ingest": link(request, "ingest", method="POST"),
            "api_root": link(request, "api_root"),
        },
    )
