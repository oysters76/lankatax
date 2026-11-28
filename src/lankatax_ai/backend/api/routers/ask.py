"""Answer tax questions with retrieval-augmented generation.

The endpoint is public; anonymous callers are rate limited per client address.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from lankatax_ai.backend.api.deps import ApiDepsDep
from lankatax_ai.backend.api.schemas import (
    AskRequestSchema,
    AskResponse,
    AskUsageSchema,
    CitationSchema,
)
from lankatax_ai.backend.api.utils import link
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
                chunk_id=citation.chunk_id,
                document_id=citation.document_id,
                title=citation.title,
                url=citation.url,
                snippet=citation.snippet,
                score=citation.score,
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
    client_key = request.client.host if request.client is not None else "anonymous"
    outcome = await run_ask(
        AskRequest(
            question=payload.question,
            conversation_id=payload.conversation_id,
            fingerprint=client_key,
        ),
        deps=deps.ask,
    )
    return _response(request, outcome)
