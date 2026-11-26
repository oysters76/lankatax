"""Search over the tax corpus; the use case does the work."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from lankatax_ai.backend.api.deps import ApiDepsDep
from lankatax_ai.backend.api.schemas import SearchRequest, SearchResponse
from lankatax_ai.backend.api.security import require_access
from lankatax_ai.backend.api.utils import to_search_options, to_search_response
from lankatax_ai.backend.services.search import search as run_search

router = APIRouter(tags=["search"])


@router.post(
    "/api/v1/search",
    response_model=SearchResponse,
    summary="Search the tax corpus",
    dependencies=[Depends(require_access)],
)
async def search(request: Request, payload: SearchRequest, deps: ApiDepsDep) -> SearchResponse:
    outcome = await run_search(
        payload.query,
        to_search_options(payload),
        session_factory=deps.session_factory,
        deps=deps.search,
    )
    return to_search_response(request, payload, outcome)
