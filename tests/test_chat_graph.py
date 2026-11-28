from __future__ import annotations

from uuid import uuid4

import pytest

from lankatax_ai.backend.analytics.types import SearchTiming
from lankatax_ai.backend.chat import (
    AnswerDraft,
    CondensedQuestion,
    GraphConfig,
    GraphDeps,
    build_ask_graph,
)
from lankatax_ai.backend.guardrails import (
    GroundingCheck,
    GuardClassification,
    GuardrailReason,
    InputValidator,
    OutputChecker,
    OutputVerdict,
)
from lankatax_ai.backend.rag.types import PreprocessedQuery, RetrievalResult
from lankatax_ai.backend.services.search.types import SearchOutcome
from lankatax_ai.backend.types import QueryType
from tests.helpers import FakeStructuredLLM

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


def _outcome(results: list[RetrievalResult], context: str) -> SearchOutcome:
    return SearchOutcome(
        processed_query=PreprocessedQuery(original="q", expanded="q", query_type=QueryType.factual),
        results=results,
        timings=SearchTiming(),
        context=context,
    )


def _graph(
    responses: dict[type, object],
    search,
    *,
    min_score: float = 0.2,
    guards: bool = True,
):
    fake = FakeStructuredLLM(responses)
    return (
        build_ask_graph(
            GraphDeps(
                llm=fake,  # type: ignore[arg-type]
                search=search,
                config=GraphConfig(min_score=min_score, guard_input=guards, guard_output=guards),
                input_validator=InputValidator(llm=fake) if guards else None,
                output_checker=OutputChecker(llm=fake) if guards else None,
            )
        ),
        fake,
    )


async def test_happy_path_returns_a_grounded_answer():
    queries: list[str] = []

    async def search(query: str) -> SearchOutcome:
        queries.append(query)
        return _outcome([_result()], "[1] PAYE is withheld by employers every month.")

    graph, _ = _graph(
        {
            GuardClassification: GuardClassification(allowed=True),
            AnswerDraft: AnswerDraft(answer="PAYE is withheld monthly [1].", citation_numbers=[1]),
            GroundingCheck: GroundingCheck(grounded=True),
        },
        search,
    )

    state = await graph.ainvoke({"question": "How is PAYE withheld?", "history": [], "attempts": 0})

    assert state["refused"] is False
    assert state["answer"] == "PAYE is withheld monthly [1]."
    assert state["output_guard"].verdict is OutputVerdict.ok
    assert len(state["retrieved"]) == 1
    assert queries == ["How is PAYE withheld?"]
    assert state["timings"]["retrieve"] >= 0


async def test_input_refusal_skips_retrieval():
    called = False

    async def search(query: str) -> SearchOutcome:
        nonlocal called
        called = True
        return _outcome([], "")

    graph, _ = _graph(
        {
            GuardClassification: GuardClassification(
                allowed=False, category="off_topic", reason="weather"
            )
        },
        search,
    )

    state = await graph.ainvoke({"question": "What's the weather?", "history": [], "attempts": 0})

    assert state["refused"] is True
    assert state["reason"] is GuardrailReason.off_topic
    assert "Sri Lankan taxation" in state["answer"]
    assert called is False


async def test_no_results_refuses_with_insufficient_context():
    async def search(query: str) -> SearchOutcome:
        return _outcome([], "")

    graph, _ = _graph(
        {GuardClassification: GuardClassification(allowed=True)},
        search,
    )

    state = await graph.ainvoke(
        {"question": "Capital gains on land?", "history": [], "attempts": 0}
    )

    assert state["refused"] is True
    assert state["reason"] is GuardrailReason.insufficient_context


async def test_low_similarity_is_refused():
    async def search(query: str) -> SearchOutcome:
        return _outcome([_result(score=0.05)], "[1] unrelated text")

    graph, _ = _graph(
        {GuardClassification: GuardClassification(allowed=True)},
        search,
        min_score=0.2,
    )

    state = await graph.ainvoke({"question": "Something obscure", "history": [], "attempts": 0})

    assert state["refused"] is True
    assert state["reason"] is GuardrailReason.insufficient_context


async def test_ungrounded_answer_is_retried_once_then_accepted():
    drafts = [
        AnswerDraft(answer="PAYE is 50% [1].", citation_numbers=[1]),
        AnswerDraft(answer="PAYE is withheld monthly [1].", citation_numbers=[1]),
    ]
    groundings = [
        GroundingCheck(grounded=False, unsupported_claims=["PAYE is 50%"]),
        GroundingCheck(grounded=True),
    ]

    async def search(query: str) -> SearchOutcome:
        return _outcome([_result()], "[1] PAYE is withheld by employers every month.")

    graph, _ = _graph(
        {
            GuardClassification: GuardClassification(allowed=True),
            AnswerDraft: drafts,
            GroundingCheck: groundings,
        },
        search,
    )

    state = await graph.ainvoke({"question": "How is PAYE withheld?", "history": [], "attempts": 0})

    assert state["refused"] is False
    assert state["attempts"] == 1
    assert state["answer"] == "PAYE is withheld monthly [1]."


async def test_second_ungrounded_answer_is_refused():
    async def search(query: str) -> SearchOutcome:
        return _outcome([_result()], "[1] PAYE is withheld by employers every month.")

    graph, _ = _graph(
        {
            GuardClassification: GuardClassification(allowed=True),
            AnswerDraft: [
                AnswerDraft(answer="PAYE is 50% [1].", citation_numbers=[1]),
                AnswerDraft(answer="PAYE is 50% [1].", citation_numbers=[1]),
            ],
            GroundingCheck: [
                GroundingCheck(grounded=False),
                GroundingCheck(grounded=False),
            ],
        },
        search,
    )

    state = await graph.ainvoke({"question": "How is PAYE withheld?", "history": [], "attempts": 0})

    assert state["refused"] is True
    assert state["reason"] is GuardrailReason.ungrounded


async def test_condense_uses_history_when_present():
    from lankatax_ai.backend.memory import Turn

    queries: list[str] = []

    async def search(query: str) -> SearchOutcome:
        queries.append(query)
        return _outcome([_result()], "[1] PAYE is withheld by employers every month.")

    graph, _ = _graph(
        {
            GuardClassification: GuardClassification(allowed=True),
            CondensedQuestion: CondensedQuestion(
                question="How is PAYE withheld for company directors?"
            ),
            AnswerDraft: AnswerDraft(answer="Monthly [1].", citation_numbers=[1]),
            GroundingCheck: GroundingCheck(grounded=True),
        },
        search,
    )

    await graph.ainvoke(
        {
            "question": "What about directors?",
            "history": [Turn(role="user", content="How is PAYE withheld?")],
            "attempts": 0,
        }
    )

    assert queries == ["How is PAYE withheld for company directors?"]
