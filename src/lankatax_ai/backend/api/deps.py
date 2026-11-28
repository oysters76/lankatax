"""Service wiring for the API layer.

The API never builds adapters inside a router: everything arrives through
:class:`ApiDeps`, which ``create_app`` puts on ``app.state``.
"""

from __future__ import annotations

import dataclasses
from typing import Annotated, cast

from fastapi import Depends, Request

from lankatax_ai.backend.api.security import AuthConfig
from lankatax_ai.backend.db.session import async_session
from lankatax_ai.backend.jobs.client import build_hatchet_client
from lankatax_ai.backend.jobs.gateway import HatchetJobGateway
from lankatax_ai.backend.llm import build_embeddings
from lankatax_ai.backend.services.ask import AskDeps, build_ask_deps
from lankatax_ai.backend.services.cache import CacheDeps, build_cache_deps
from lankatax_ai.backend.services.ingest import IngestDeps, build_ingest_deps
from lankatax_ai.backend.services.jobs import JobDeps
from lankatax_ai.backend.services.protocols import SessionFactory
from lankatax_ai.backend.services.scrape import ScrapeDeps, build_scrape_deps
from lankatax_ai.backend.services.search import SearchDeps, build_search_deps
from lankatax_ai.backend.services.stats import StatsDeps, build_stats_deps

__all__ = ["ApiDeps", "ApiDepsDep", "build_api_deps", "get_api_deps"]


@dataclasses.dataclass(frozen=True, slots=True)
class ApiDeps:
    """Everything a router may need, wired by the caller."""

    session_factory: SessionFactory
    ask: AskDeps
    search: SearchDeps
    stats: StatsDeps
    cache: CacheDeps
    jobs: JobDeps
    scrape: ScrapeDeps
    ingest: IngestDeps
    auth: AuthConfig


def build_api_deps(*, auth: AuthConfig) -> ApiDeps:
    """Wire the production collaborators; called once per process."""
    embeddings = build_embeddings()
    search_deps = build_search_deps(embeddings)
    return ApiDeps(
        session_factory=async_session,
        ask=build_ask_deps(embeddings=embeddings, search_deps=search_deps),
        search=search_deps,
        stats=build_stats_deps(),
        cache=build_cache_deps(),
        jobs=JobDeps(gateway=HatchetJobGateway(build_hatchet_client())),
        scrape=build_scrape_deps(),
        ingest=build_ingest_deps(embeddings),
        auth=auth,
    )


def get_api_deps(request: Request) -> ApiDeps:
    """Read the wired dependencies from app state."""
    return cast(ApiDeps, request.app.state.api_deps)


ApiDepsDep = Annotated[ApiDeps, Depends(get_api_deps)]
