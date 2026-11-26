"""Search quality metrics over the analytics store."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from lankatax_ai.backend.api.deps import ApiDepsDep
from lankatax_ai.backend.api.schemas import StatsResponse
from lankatax_ai.backend.api.security import require_access
from lankatax_ai.backend.api.utils import link
from lankatax_ai.backend.services.stats import get_stats

router = APIRouter(tags=["stats"])


@router.get(
    "/api/v1/stats",
    response_model=StatsResponse,
    summary="Search quality metrics",
    dependencies=[Depends(require_access)],
)
async def stats(request: Request, deps: ApiDepsDep) -> StatsResponse:
    result = await get_stats(session_factory=deps.session_factory, deps=deps.stats)
    return StatsResponse(
        total_queries=result.total_queries,
        avg_latency_ms=result.avg_latency_ms,
        hit_rate=result.hit_rate,
        no_result_queries=result.no_result_queries,
        avg_top_score=result.avg_top_score,
        cache_hit_count=result.cache_hit_count,
        multi_query_count=result.multi_query_count,
        links={
            "self": link(request, "stats"),
            "search": link(request, "search", method="POST"),
            "cache_stats": link(request, "cache_stats"),
            "api_root": link(request, "api_root"),
        },
    )
