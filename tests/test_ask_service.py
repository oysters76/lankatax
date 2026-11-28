from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

import pytest

from lankatax_ai.backend.chat import AnswerDraft
from lankatax_ai.backend.config import settings
from lankatax_ai.backend.guardrails import GuardrailReason, RateLimiter
from lankatax_ai.backend.memory import InMemoryConversationStore, Turn
from lankatax_ai.backend.rag.types import RetrievalResult
from lankatax_ai.backend.services.ask import AskDeps, AskOptions, AskRequest, ask
from lankatax_ai.backend.services.errors import RateLimitError, UpstreamError, ValidationError

pytestmark = pytest.mark.unit


def _result(score: float = 0.8) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content="PAYE is withheld by employers every month.",
        score=score,
        vector_score=score,
        keyword_score=None,
        metadata={"document_title": "PAYE guide", "from_url": "https://ird.gov.lk/paye"},
    )


class _FakeGraph:
    def __init__(self, state: Mapping[str, Any]) -> None:
        self._state = state
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls.append(dict(state))
        return {**state, **self._state}


class _SlowGraph:
    async def ainvoke(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        await asyncio.sleep(0.05)
        return dict(state)


def _deps(graph: Any, *, store: Any = None, limiter: Any = None) -> AskDeps:
    return AskDeps(
        graph_factory=lambda options: graph,
        conversation_store=store or InMemoryConversationStore(),
        rate_limiter=limiter or RateLimiter(limit=100),
    )


def _answer_state(**overrides: Any) -> dict[str, Any]:
    state = {
        "question": "How is PAYE withheld?",
        "answer": "Employers withhold it monthly [1].",
        "refused": False,
        "reason": None,
        "draft": AnswerDraft(answer="Employers withhold it monthly [1].", citation_numbers=[1]),
        "retrieved": [_result()],
        "timings": {"answer": 4, "retrieve": 7},
        "usage": {"llm_calls": 3},
    }
    return state | overrides


async def test_ask_returns_answer_and_citations_and_persists():
    graph = _FakeGraph(_answer_state())
    store = InMemoryConversationStore()
    deps = _deps(graph, store=store)

    outcome = await ask(AskRequest(question="How is PAYE withheld?"), deps=deps)

    assert outcome.answer == "Employers withhold it monthly [1]."
    assert outcome.refused is False
    assert outcome.citations[0].title == "PAYE guide"
    assert outcome.citations[0].url == "https://ird.gov.lk/paye"
    assert outcome.usage.llm_calls == 3
    assert outcome.usage.retrieved_chunks == 1
    assert outcome.timings == {"answer": 4, "retrieve": 7}
    assert graph.calls[0]["history"] == []
    turns = await store.load(outcome.conversation_id, limit=10)
    assert [turn.role for turn in turns] == ["user", "assistant"]
    assert turns[1].sources[0]["title"] == "PAYE guide"


async def test_refusal_is_not_persisted():
    graph = _FakeGraph(
        _answer_state(
            answer="I can only answer questions about Sri Lankan taxation.",
            refused=True,
            reason=GuardrailReason.off_topic,
            draft=None,
            retrieved=[],
        )
    )
    store = InMemoryConversationStore()
    deps = _deps(graph, store=store)

    outcome = await ask(AskRequest(question="What's the weather?"), deps=deps)

    assert outcome.refused is True
    assert outcome.reason is GuardrailReason.off_topic
    assert outcome.citations == ()
    assert await store.load(outcome.conversation_id, limit=10) == []


async def test_existing_conversation_history_reaches_the_graph():
    store = InMemoryConversationStore()
    conversation_id = await store.create(fingerprint="user")
    await store.append(conversation_id, [Turn(role="user", content="How is PAYE withheld?")])
    graph = _FakeGraph(_answer_state())
    deps = _deps(graph, store=store)

    outcome = await ask(
        AskRequest(question="What about directors?", conversation_id=conversation_id),
        deps=deps,
    )

    assert outcome.conversation_id == conversation_id
    history = graph.calls[0]["history"]
    assert [turn.content for turn in history] == ["How is PAYE withheld?"]
    turns = await store.load(conversation_id, limit=10)
    assert len(turns) == 3


async def test_empty_question_is_a_validation_error():
    deps = _deps(_FakeGraph(_answer_state()))

    with pytest.raises(ValidationError, match="must not be empty"):
        await ask(AskRequest(question="   "), deps=deps)


async def test_rate_limit_is_enforced_before_the_graph_runs():
    graph = _FakeGraph(_answer_state())
    deps = _deps(graph, limiter=RateLimiter(limit=1))

    await ask(AskRequest(question="first"), deps=deps)
    with pytest.raises(RateLimitError):
        await ask(AskRequest(question="second"), deps=deps)

    assert len(graph.calls) == 1


async def test_timeout_maps_to_an_upstream_error(monkeypatch):
    monkeypatch.setattr(settings, "ask_timeout_seconds", 0.001)
    deps = _deps(_SlowGraph())

    with pytest.raises(UpstreamError, match="too long"):
        await ask(AskRequest(question="How is PAYE withheld?"), deps=deps)


async def test_unknown_citation_numbers_are_dropped():
    graph = _FakeGraph(
        _answer_state(
            draft=AnswerDraft(answer="PAYE is 18% [3].", citation_numbers=[3]),
        )
    )
    deps = _deps(graph)

    outcome = await ask(AskRequest(question="PAYE rate?"), deps=deps)

    assert outcome.citations == ()


async def test_options_are_validated():
    with pytest.raises(ValidationError, match="Top-k"):
        AskOptions(top_k=0)
    with pytest.raises(ValidationError, match="min_score"):
        AskOptions(min_score=2.0)
