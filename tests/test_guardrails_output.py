from __future__ import annotations

import pytest

from lankatax_ai.backend.guardrails import (
    GroundingCheck,
    GuardrailReason,
    OutputChecker,
    OutputVerdict,
)
from tests.helpers import FakeStructuredLLM

pytestmark = pytest.mark.unit


def _checker(grounding: GroundingCheck | None = None) -> tuple[OutputChecker, FakeStructuredLLM]:
    fake = FakeStructuredLLM({GroundingCheck: grounding} if grounding else None)
    return OutputChecker(llm=fake), fake  # type: ignore[arg-type]


async def test_empty_answer_is_refused():
    checker, _ = _checker()

    result = await checker.check(question="q", answer="  ", context="c", citation_count=1)

    assert result.verdict is OutputVerdict.refuse
    assert result.reason is GuardrailReason.empty


async def test_unknown_citation_triggers_a_retry():
    checker, _ = _checker(GroundingCheck(grounded=True))

    result = await checker.check(
        question="q",
        answer="PAYE is withheld monthly [3].",
        context="[1] PAYE ...",
        citation_count=2,
    )

    assert result.verdict is OutputVerdict.retry
    assert result.reason is GuardrailReason.ungrounded
    assert "3" in result.detail


async def test_raw_html_is_refused():
    checker, _ = _checker(GroundingCheck(grounded=True))

    result = await checker.check(
        question="q",
        answer="<script>alert(1)</script>",
        context="c",
        citation_count=1,
    )

    assert result.verdict is OutputVerdict.refuse
    assert result.reason is GuardrailReason.unsafe


async def test_ungrounded_answer_triggers_a_retry():
    checker, _ = _checker(
        GroundingCheck(
            grounded=False,
            unsupported_claims=["PAYE is 50%"],
            reason="context says 18%",
        )
    )

    result = await checker.check(
        question="q",
        answer="PAYE is 50% [1].",
        context="[1] PAYE is 18%.",
        citation_count=1,
    )

    assert result.verdict is OutputVerdict.retry
    assert result.unsupported == ("PAYE is 50%",)


async def test_grounded_answer_passes_and_masks_pii():
    checker, _ = _checker(GroundingCheck(grounded=True))

    result = await checker.check(
        question="q",
        answer="Write to tax@example.com about PAYE [1].",
        context="[1] PAYE ...",
        citation_count=1,
    )

    assert result.verdict is OutputVerdict.ok
    assert "tax@example.com" not in result.answer
    assert "[redacted-email]" in result.answer


async def test_classifier_failure_accepts_the_answer():
    checker, fake = _checker(GroundingCheck(grounded=True))
    fake.error = RuntimeError("provider down")

    result = await checker.check(
        question="q",
        answer="PAYE is withheld [1].",
        context="[1] PAYE ...",
        citation_count=1,
    )

    assert result.verdict is OutputVerdict.ok
