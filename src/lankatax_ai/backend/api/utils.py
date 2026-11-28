"""FastAPI-specific helpers: link building and service mappers.

Routing policy lives here too: the route-name set below decides whether a link
advertises ``auth_required``, so a route is described in exactly one place.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from lankatax_ai.backend.api.schemas import (
    ErrorBody,
    JobReceiptSchema,
    Link,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    TimingModel,
)
from lankatax_ai.backend.rag.metadata_filter import MetadataFilters
from lankatax_ai.backend.rag.types import RetrievalResult
from lankatax_ai.backend.services.jobs import JobReceipt
from lankatax_ai.backend.services.search import SearchOptions, SearchOutcome
from lankatax_ai.backend.types import SearchSource

__all__ = [
    "DOCS_ROUTES",
    "PUBLIC_ROUTES",
    "error_payload",
    "job_receipt_response",
    "link",
    "to_search_options",
    "to_search_response",
]

#: Routes that do not need a bearer token.
PUBLIC_ROUTES: frozenset[str] = frozenset({"root", "health", "api_root", "token", "ask"})

#: Documentation routes, guarded by their own setting.
DOCS_ROUTES: frozenset[str] = frozenset({"swagger_ui_html", "redoc_html", "openapi"})


def link(
    request: Request,
    name: str,
    *,
    method: str = "GET",
    title: str | None = None,
    **params: Any,
) -> Link:
    """Build one link from a route name, never from a literal path."""
    href = str(request.url_for(name, **params))
    auth_required = name not in PUBLIC_ROUTES
    if name in DOCS_ROUTES:
        auth_required = bool(getattr(request.app.state, "docs_protected", True))
    return Link(
        href=href,
        method=method,
        title=title,
        auth_required=auth_required,
    )


def job_receipt_response(request: Request, receipt: JobReceipt) -> JSONResponse:
    """Render an accepted job with Location and idempotent-replay headers."""
    detail = link(request, "job_detail", run_id=receipt.run_id)
    body = JobReceiptSchema(
        run_id=receipt.run_id,
        reused=receipt.reused,
        links={"self": detail, "api_root": link(request, "api_root")},
    )
    headers = {"Location": detail.href}
    if receipt.reused:
        headers["Idempotent-Replayed"] = "true"
    return JSONResponse(
        status_code=202,
        content=body.model_dump(mode="json"),
        headers=headers,
    )


def error_payload(
    request: Request, *, code: str, detail: str, errors: list[dict[str, Any]] | None = None
) -> ErrorBody:
    """The single error body; 401s also advertise the token endpoint."""
    links = {"api_root": link(request, "api_root")}
    if code == "unauthorized":
        links["token"] = link(request, "token", method="POST")
    return ErrorBody(code=code, detail=detail, errors=errors, links=links)


def to_search_options(payload: SearchRequest) -> SearchOptions:
    """Translate the request body into service options; the source is ours."""
    return SearchOptions(
        source=SearchSource.api,
        mode=payload.mode,
        filters=MetadataFilters(
            category=payload.category,
            doc_type=payload.doc_type,
            after=payload.after,
            before=payload.before,
        ),
        top_k=payload.top_k,
        rerank=payload.rerank,
        multi_query=payload.multi_query,
        use_cache=payload.use_cache,
        include_context=payload.include_context,
        vector_weight=payload.vector_weight,
        rerank_threshold=payload.rerank_threshold,
        dedup_threshold=payload.dedup_threshold,
        token_budget=payload.token_budget,
        llm_provider=payload.llm_provider,
    )


def _result_item(result: RetrievalResult) -> SearchResultItem:
    url = result.metadata.get("url") or result.metadata.get("from_url")
    item_links: dict[str, Link] = {}
    if isinstance(url, str):
        title = result.metadata.get("document_title") or result.metadata.get("title")
        item_links["source"] = Link(
            href=url,
            method="GET",
            title=title if isinstance(title, str) else None,
        )
    return SearchResultItem(
        chunk_id=result.chunk_id,
        document_id=result.document_id,
        content=result.content,
        score=result.score,
        vector_score=result.vector_score,
        keyword_score=result.keyword_score,
        metadata=result.metadata,
        links=item_links,
    )


def to_search_response(
    request: Request, payload: SearchRequest, outcome: SearchOutcome
) -> SearchResponse:
    """Map the service outcome onto the HTTP schema (never leak ``embedding``)."""
    timings = outcome.timings
    return SearchResponse(
        query=payload.query,
        expanded_query=outcome.processed_query.expanded,
        query_type=outcome.processed_query.query_type,
        results=[_result_item(result) for result in outcome.results],
        timings=TimingModel(
            embed_ms=timings.embed_ms,
            search_ms=timings.search_ms,
            rerank_ms=timings.rerank_ms,
            total_ms=timings.total_ms,
        ),
        cache_hit=outcome.cache_hit,
        multi_query=outcome.multi_query,
        context=outcome.context,
        links={
            "self": link(request, "search", method="POST"),
            "stats": link(request, "stats"),
            "cache_stats": link(request, "cache_stats"),
            "api_root": link(request, "api_root"),
        },
    )
