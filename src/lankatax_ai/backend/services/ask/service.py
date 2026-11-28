"""The ask use case: rate limit, load memory, run the graph, persist the turn.

Everything external (LLM, retrieval, storage) arrives through ``AskDeps``; this
module owns only the flow and the transport-neutral outcome.
"""

from __future__ import annotations

import asyncio
import logging
import re

from lankatax_ai.backend.config import settings
from lankatax_ai.backend.memory.types import Turn
from lankatax_ai.backend.rag.types import RetrievalResult
from lankatax_ai.backend.services.ask.metrics import record_ask
from lankatax_ai.backend.services.ask.types import (
    AskDeps,
    AskOutcome,
    AskRequest,
    AskUsage,
    Citation,
)
from lankatax_ai.backend.services.errors import (
    GatewayTimeoutError,
    RateLimitError,
    ServiceError,
    UpstreamError,
    ValidationError,
)

__all__ = ["ask"]

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")
_ANONYMOUS = "anonymous"


async def ask(request: AskRequest, *, deps: AskDeps) -> AskOutcome:
    """Answer one question, persisting the exchange on success."""
    question = request.question.strip()
    if not question:
        raise ValidationError("Question must not be empty")

    owner = request.requested_by or request.fingerprint or _ANONYMOUS
    # Quota first: it protects conversation lookups and the LLM, so even an
    # invalid conversation id costs the caller a request.
    await deps.rate_limiter.check(owner)

    conversation_id = request.conversation_id
    if conversation_id is None:
        conversation_id = await deps.conversation_store.create(fingerprint=owner)

    history = await deps.conversation_store.load(
        conversation_id, limit=request.options.memory_turns, owner=owner
    )

    graph = deps.graph_factory(request.options)
    initial: dict[str, object] = {
        "question": question,
        "conversation_id": conversation_id,
        "history": history,
        "attempts": 0,
        "usage": {},
        "timings": {},
    }

    if deps.slots is not None and deps.slots.locked():
        raise RateLimitError(
            "The answer service is busy; retry shortly",
            retry_after=5,
        )

    try:
        async with asyncio.timeout(settings.ask_timeout_seconds):
            if deps.slots is None:
                state = dict(await graph.ainvoke(initial))
            else:
                async with deps.slots:
                    state = dict(await graph.ainvoke(initial))
    except TimeoutError as error:
        raise GatewayTimeoutError("The answer took too long to generate") from error
    except ServiceError:
        raise
    except Exception as error:
        logger.exception("Ask graph failed")
        raise UpstreamError("The answer provider failed") from error

    refused = bool(state.get("refused"))
    answer = str(state.get("answer") or "")
    reason = state.get("reason")
    retrieved = tuple(state.get("retrieved") or ())
    citations = () if refused else _resolve_citations(state, retrieved)

    outcome = AskOutcome(
        conversation_id=conversation_id,
        answer=answer,
        citations=citations,
        refused=refused,
        reason=reason if refused else None,
        usage=AskUsage(
            llm_calls=int((state.get("usage") or {}).get("llm_calls", 0)),
            answer_chars=len(answer),
            retrieved_chunks=len(retrieved),
        ),
        timings=dict(state.get("timings") or {}),
        retrieved=retrieved,
    )

    if not refused:
        await deps.conversation_store.append(
            conversation_id,
            [
                Turn(role="user", content=str(state.get("question") or question)),
                Turn(
                    role="assistant",
                    content=answer,
                    sources=tuple(_source(citation) for citation in citations),
                ),
            ],
            owner=owner,
        )

    record_ask(outcome)
    return outcome


def _resolve_citations(
    state: dict[str, object],
    retrieved: tuple[RetrievalResult, ...],
) -> tuple[Citation, ...]:
    numbers = _cited_numbers(state)
    seen: set[int] = set()
    citations: list[Citation] = []

    for number in numbers:
        if number in seen or not 1 <= number <= len(retrieved):
            continue
        result = retrieved[number - 1]
        citations.append(
            Citation(
                chunk_id=str(result.chunk_id),
                document_id=str(result.document_id),
                title=str(result.metadata.get("document_title") or "Untitled"),
                url=_metadata_url(result.metadata),
                snippet=result.content[:280],
                score=result.score,
                page_start=_page_number(result.metadata.get("page_start")),
                page_end=_page_number(result.metadata.get("page_end")),
            )
        )
        seen.add(number)

    return tuple(citations)


def _page_number(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _metadata_url(metadata: dict[str, object]) -> str | None:
    value = metadata.get("url") or metadata.get("from_url")
    return value if isinstance(value, str) else None


def _cited_numbers(state: dict[str, object]) -> list[int]:
    draft = state.get("draft")
    declared = getattr(draft, "citation_numbers", None)
    if declared:
        return [int(number) for number in declared]
    answer = str(state.get("answer") or "")
    return sorted({int(match) for match in _CITATION_RE.findall(answer)})


def _source(citation: Citation) -> dict[str, object]:
    return {
        "chunk_id": citation.chunk_id,
        "document_id": citation.document_id,
        "title": citation.title,
        "url": citation.url,
    }
