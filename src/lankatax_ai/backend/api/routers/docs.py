"""OpenAPI schema and the two HTML viewers, rebuilt so they can be guarded.

FastAPI's built-in docs routes are disabled in ``create_app``; these replacements
behave identically but can carry the access guard.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse

router = APIRouter(tags=["discovery"], include_in_schema=False)


@router.get("/openapi.json", include_in_schema=False, name="openapi")
async def openapi(request: Request) -> JSONResponse:
    return JSONResponse(request.app.openapi())


@router.get("/docs", include_in_schema=False, name="swagger_ui_html")
async def swagger_ui_html(request: Request) -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=f"{request.app.title} — Swagger UI",
    )


@router.get("/redoc", include_in_schema=False, name="redoc_html")
async def redoc_html(request: Request) -> HTMLResponse:
    return get_redoc_html(
        openapi_url="/openapi.json",
        title=f"{request.app.title} — ReDoc",
    )
