"""Scrape job and status endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from lankatax_ai.backend.api.deps import ApiDepsDep
from lankatax_ai.backend.api.schemas import (
    JobReceiptSchema,
    RegistryEntrySchema,
    RegistrySchema,
    ScrapeRequest,
    ScrapeStatusSchema,
)
from lankatax_ai.backend.api.security import require_access
from lankatax_ai.backend.api.utils import job_receipt_response, link
from lankatax_ai.backend.services.jobs import JobRequest, JobType, enqueue_job
from lankatax_ai.backend.services.scrape import get_scrape_status

router = APIRouter(tags=["scrape"])

_PROTECTED = [Depends(require_access)]


@router.post(
    "/api/v1/scrape",
    response_model=JobReceiptSchema,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Scrape the document registry",
    dependencies=_PROTECTED,
)
async def scrape(request: Request, payload: ScrapeRequest, deps: ApiDepsDep) -> JSONResponse:
    params = payload.model_dump(exclude={"idempotency_key"}, exclude_none=True)
    if payload.idempotency_key:
        params["idempotency_key"] = payload.idempotency_key
    receipt = await enqueue_job(
        JobRequest(
            type=JobType.scrape_registry,
            params=params,
            idempotency_key=payload.idempotency_key,
        ),
        deps=deps.jobs,
    )
    return job_receipt_response(request, receipt)


@router.get(
    "/api/v1/scrape/urls",
    response_model=RegistrySchema,
    summary="Registry entries",
    dependencies=_PROTECTED,
)
async def scrape_urls(request: Request, deps: ApiDepsDep) -> RegistrySchema:
    return RegistrySchema(
        items=[
            RegistryEntrySchema(
                url=entry.url,
                category=entry.category.value,
                label=entry.label,
                kind=entry.kind.value,
            )
            for entry in deps.scrape.registry.get_all()
        ],
        links={
            "self": link(request, "scrape_urls"),
            "scrape": link(request, "scrape", method="POST"),
            "api_root": link(request, "api_root"),
        },
    )


@router.get(
    "/api/v1/scrape/status",
    response_model=ScrapeStatusSchema,
    summary="Scrape status",
    dependencies=_PROTECTED,
)
async def scrape_status(request: Request, deps: ApiDepsDep) -> ScrapeStatusSchema:
    record = await get_scrape_status(
        session_factory=deps.session_factory,
        deps=deps.scrape,
    )
    return ScrapeStatusSchema(
        total=record.total,
        recent=record.recent,
        first_scrape=record.first_scrape,
        last_scrape=record.last_scrape,
        links={
            "self": link(request, "scrape_status"),
            "scrape": link(request, "scrape", method="POST"),
            "scrape_urls": link(request, "scrape_urls"),
            "api_root": link(request, "api_root"),
        },
    )
