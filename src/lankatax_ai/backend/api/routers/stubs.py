"""Declared but unimplemented actions."""

from __future__ import annotations

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from lankatax_ai.backend.api.utils import not_implemented

router = APIRouter(tags=["platform"])

_CHAT_PHASE = "the chat service phase"


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
