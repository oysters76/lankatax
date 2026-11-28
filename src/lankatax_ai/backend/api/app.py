"""Application factory: routers, error handlers, lifespan and metadata."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from lankatax_ai.backend.api.deps import ApiDeps, build_api_deps
from lankatax_ai.backend.api.errors import install_error_handlers
from lankatax_ai.backend.api.routers import (
    auth,
    cache,
    docs,
    ingest,
    jobs,
    pipeline,
    root,
    scrape,
    search,
    stats,
    stubs,
)
from lankatax_ai.backend.api.security import AuthConfig, require_access
from lankatax_ai.backend.config import settings
from lankatax_ai.backend.db.session import dispose_engine

__all__ = ["create_app"]

TITLE = "LankaTax API"
VERSION = "0.1.0"
DESCRIPTION = (
    "Retrieval-augmented answers over Sri Lankan tax documents.\n\n"
    "This API uses hypermedia controls (Richardson Maturity Level 3): start at "
    "`/api/v1` and follow the `links` in every response instead of building URLs "
    "yourself."
)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    await dispose_engine()


def create_app(
    *,
    api_deps: ApiDeps | None = None,
    auth_config: AuthConfig | None = None,
    docs_require_auth: bool | None = None,
) -> FastAPI:
    """Build the app; tests pass fakes, production reads settings."""
    resolved_auth = auth_config or AuthConfig.from_settings(settings)
    resolved_deps = api_deps or build_api_deps(auth=resolved_auth)
    protected_docs = (
        settings.api_docs_require_auth if docs_require_auth is None else docs_require_auth
    )

    app = FastAPI(
        title=TITLE,
        version=VERSION,
        description=DESCRIPTION,
        lifespan=_lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.api_deps = resolved_deps
    app.state.auth = resolved_auth
    app.state.docs_protected = protected_docs

    install_error_handlers(app)
    app.include_router(root.router)
    app.include_router(auth.router)
    app.include_router(search.router)
    app.include_router(stats.router)
    app.include_router(cache.router)
    app.include_router(jobs.router)
    app.include_router(scrape.router)
    app.include_router(ingest.router)
    app.include_router(pipeline.router)
    app.include_router(stubs.router)
    app.include_router(
        docs.router,
        dependencies=[Depends(require_access)] if protected_docs else [],
    )
    return app
