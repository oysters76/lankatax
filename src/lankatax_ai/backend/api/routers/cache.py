"""Semantic cache administration: stats, clear, expire."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from lankatax_ai.backend.api.deps import ApiDepsDep
from lankatax_ai.backend.api.schemas import CacheResponse, CountResponse
from lankatax_ai.backend.api.security import require_access
from lankatax_ai.backend.api.utils import link
from lankatax_ai.backend.services.cache import (
    clear_cache,
    get_cache_stats,
    invalidate_expired,
)

router = APIRouter(tags=["cache"])

_PROTECTED = [Depends(require_access)]


@router.get(
    "/api/v1/cache",
    response_model=CacheResponse,
    summary="Cache statistics",
    dependencies=_PROTECTED,
)
async def cache_stats(request: Request, deps: ApiDepsDep) -> CacheResponse:
    stats = await get_cache_stats(session_factory=deps.session_factory, deps=deps.cache)
    return CacheResponse(
        total_entries=stats.total_entries,
        total_hits=stats.total_hits,
        hit_rate=stats.hit_rate,
        oldest_entry_hours=stats.oldest_entry_hours,
        links={
            "self": link(request, "cache_stats"),
            "cache_clear": link(request, "cache_clear", method="DELETE", title="Clear the cache"),
            "cache_expire": link(
                request, "cache_expire", method="POST", title="Expire stale entries"
            ),
            "search": link(request, "search", method="POST"),
            "api_root": link(request, "api_root"),
        },
    )


@router.delete(
    "/api/v1/cache",
    response_model=CountResponse,
    summary="Delete every cache entry",
    dependencies=_PROTECTED,
)
async def cache_clear(request: Request, deps: ApiDepsDep) -> CountResponse:
    deleted = await clear_cache(session_factory=deps.session_factory, deps=deps.cache)
    return CountResponse(
        deleted=deleted,
        links={
            "cache_stats": link(request, "cache_stats"),
            "search": link(request, "search", method="POST"),
            "api_root": link(request, "api_root"),
        },
    )


@router.post(
    "/api/v1/cache/expire",
    response_model=CountResponse,
    summary="Remove entries past their TTL",
    dependencies=_PROTECTED,
)
async def cache_expire(request: Request, deps: ApiDepsDep) -> CountResponse:
    removed = await invalidate_expired(session_factory=deps.session_factory, deps=deps.cache)
    return CountResponse(
        deleted=removed,
        links={
            "cache_stats": link(request, "cache_stats"),
            "search": link(request, "search", method="POST"),
            "api_root": link(request, "api_root"),
        },
    )
