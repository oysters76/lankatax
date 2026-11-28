"""Discovery entry points: the service root, health, and the API root."""

from __future__ import annotations

from fastapi import APIRouter, Request
from sqlalchemy import text

from lankatax_ai.backend.api.deps import ApiDepsDep
from lankatax_ai.backend.api.errors import ApiError
from lankatax_ai.backend.api.schemas import ApiRoot, Health, ServiceRoot
from lankatax_ai.backend.api.utils import link

router = APIRouter(tags=["discovery"])


@router.get("/", response_model=ServiceRoot, summary="Service root")
async def root(request: Request) -> ServiceRoot:
    return ServiceRoot(
        name="LankaTax API",
        version="0.1.0",
        description="Follow the links to discover the API.",
        links={
            "api_root": link(request, "api_root"),
            "health": link(request, "health"),
            "docs": link(request, "swagger_ui_html", title="Swagger UI"),
            "openapi": link(request, "openapi", title="OpenAPI schema"),
        },
    )


@router.get("/health", response_model=Health, summary="Liveness and database check")
async def health(request: Request, deps: ApiDepsDep) -> Health:
    try:
        async with deps.session_factory() as session:
            await session.execute(text("SELECT 1"))
    except Exception as exc:
        raise ApiError(
            status_code=503, code="unavailable", detail="Database is unavailable"
        ) from exc
    return Health(links={"self": link(request, "health"), "api_root": link(request, "api_root")})


@router.get("/api/v1", response_model=ApiRoot, summary="Every available action")
async def api_root(request: Request) -> ApiRoot:
    return ApiRoot(
        version="v1",
        links={
            "token": link(
                request,
                "token",
                method="POST",
                title="Exchange credentials for a token",
            ),
            "ask": link(
                request,
                "ask",
                method="POST",
                title="Ask a tax question",
            ),
            "search": link(request, "search", method="POST", title="Search the tax corpus"),
            "stats": link(request, "stats", title="Search quality metrics"),
            "cache_stats": link(request, "cache_stats", title="Cache statistics"),
            "cache_clear": link(request, "cache_clear", method="DELETE", title="Clear the cache"),
            "cache_expire": link(
                request, "cache_expire", method="POST", title="Expire stale entries"
            ),
            "ingest": link(request, "ingest", method="POST", title="Ingest documents"),
            "ingest_status": link(request, "ingest_status", title="Ingestion status"),
            "scrape": link(request, "scrape", method="POST", title="Scrape the registry"),
            "scrape_urls": link(request, "scrape_urls", title="Registry entries"),
            "scrape_status": link(request, "scrape_status", title="Scrape status"),
            "pipeline_refresh": link(
                request,
                "pipeline_refresh",
                method="POST",
                title="Refresh the corpus",
            ),
            "jobs": link(request, "jobs", title="Job runs and schedules"),
            "health": link(request, "health"),
            "docs": link(request, "swagger_ui_html", title="Swagger UI"),
        },
    )
