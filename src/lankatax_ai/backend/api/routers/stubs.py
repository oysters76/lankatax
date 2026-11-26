"""Declared but unimplemented actions.

Each answers 501 with the standard envelope. The access guard runs first, so the
eventual contract is visible — and testable — today.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from lankatax_ai.backend.api.security import require_access
from lankatax_ai.backend.api.utils import not_implemented

router = APIRouter(tags=["platform"])

_PROTECTED = [Depends(require_access)]

_CHAT_PHASE = "the chat service phase"
_JOB_PHASE = "the background job phase"


@router.post(
    "/api/v1/ask",
    summary="Ask a tax question (not implemented yet)",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
)
async def ask(request: Request) -> JSONResponse:
    return not_implemented(
        request,
        name="ask",
        phase=_CHAT_PHASE,
        method="POST",
        search="search",
        docs="swagger_ui_html",
    )


@router.post(
    "/api/v1/ingest",
    summary="Ingest documents (not implemented yet)",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    dependencies=_PROTECTED,
)
async def ingest(request: Request) -> JSONResponse:
    return not_implemented(
        request,
        name="ingest",
        phase=_JOB_PHASE,
        method="POST",
        ingest_status="ingest_status",
        scrape="scrape",
    )


@router.get(
    "/api/v1/ingest/status",
    summary="Ingestion status (not implemented yet)",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    dependencies=_PROTECTED,
)
async def ingest_status(request: Request) -> JSONResponse:
    return not_implemented(
        request,
        name="ingest_status",
        phase=_JOB_PHASE,
        ingest="ingest",
        scrape="scrape",
    )


@router.get(
    "/api/v1/scrape/urls",
    summary="Registry entries (not implemented yet)",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    dependencies=_PROTECTED,
)
async def scrape_urls(request: Request) -> JSONResponse:
    return not_implemented(
        request,
        name="scrape_urls",
        phase=_JOB_PHASE,
        scrape="scrape",
        scrape_status="scrape_status",
    )


@router.post(
    "/api/v1/scrape",
    summary="Scrape the registry (not implemented yet)",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    dependencies=_PROTECTED,
)
async def scrape(request: Request) -> JSONResponse:
    return not_implemented(
        request,
        name="scrape",
        phase=_JOB_PHASE,
        method="POST",
        scrape_status="scrape_status",
        scrape_urls="scrape_urls",
        ingest="ingest",
    )


@router.get(
    "/api/v1/scrape/status",
    summary="Scrape status (not implemented yet)",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    dependencies=_PROTECTED,
)
async def scrape_status(request: Request) -> JSONResponse:
    return not_implemented(
        request,
        name="scrape_status",
        phase=_JOB_PHASE,
        scrape="scrape",
        scrape_urls="scrape_urls",
    )


@router.get(
    "/api/v1/jobs",
    summary="Scheduled jobs (not implemented yet)",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    dependencies=_PROTECTED,
)
async def jobs(request: Request) -> JSONResponse:
    return not_implemented(
        request,
        name="jobs",
        phase=_JOB_PHASE,
        scrape="scrape",
        ingest="ingest",
    )
