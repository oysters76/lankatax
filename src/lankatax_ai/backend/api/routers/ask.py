"""Answer tax questions with retrieval-augmented generation.

The endpoint is public; anonymous callers are rate limited per client address.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Request

from lankatax_ai.backend.api.client_ip import resolve_client_ip
from lankatax_ai.backend.api.deps import ApiDepsDep
from lankatax_ai.backend.api.schemas import (
    AskRequestSchema,
    AskResponse,
    AskUsageSchema,
    CitationSchema,
)
from lankatax_ai.backend.api.utils import link
from lankatax_ai.backend.config import settings
from lankatax_ai.backend.services.ask import AskOutcome, AskRequest
from lankatax_ai.backend.services.ask import ask as run_ask

router = APIRouter(tags=["ask"])


def _response(request: Request, outcome: AskOutcome) -> AskResponse:
    return AskResponse(
        conversation_id=outcome.conversation_id,
        answer=outcome.answer,
        refused=outcome.refused,
        reason=outcome.reason.value if outcome.reason is not None else None,
        citations=[
            CitationSchema(
                chunk_id=UUID(citation.chunk_id),
                document_id=UUID(citation.document_id),
                title=citation.title,
                url=citation.url,
                snippet=citation.snippet,
                score=citation.score,
                page_start=citation.page_start,
                page_end=citation.page_end,
            )
            for citation in outcome.citations
        ],
        usage=AskUsageSchema(
            llm_calls=outcome.usage.llm_calls,
            answer_chars=outcome.usage.answer_chars,
            retrieved_chunks=outcome.usage.retrieved_chunks,
        ),
        timings=outcome.timings,
        links={
            "self": link(request, "ask", method="POST", title="Ask a tax question"),
            "search": link(request, "search", method="POST", title="Search the tax corpus"),
            "api_root": link(request, "api_root"),
        },
    )


@router.post(
    "/api/v1/ask",
    response_model=AskResponse,
    summary="Ask a tax question",
)
async def ask(
    request: Request,
    payload: AskRequestSchema,
    deps: ApiDepsDep,
) -> AskResponse:
    client_key = resolve_client_ip(
        request.client.host if request.client is not None else None,
        request.headers.get("x-forwarded-for"),
        settings.ask_trusted_proxies.split(","),
    )
    outcome = await run_ask(
        AskRequest(
            question=payload.question,
            conversation_id=(str(payload.conversation_id) if payload.conversation_id else None),
            fingerprint=client_key,
        ),
        deps=deps.ask,
    )
    return _response(request, outcome)
