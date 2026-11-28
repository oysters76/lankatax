"""Ask graph nodes: guardrails, query condensing, retrieval and answering."""

from __future__ import annotations

import logging
import time

from langchain_core.language_models import BaseChatModel

from lankatax_ai.backend.chat.constants import (
    DEFAULT_REFUSAL,
    HISTORY_CHAR_LIMIT,
    REFUSAL_MESSAGES,
    STRICT_SUFFIX,
)
from lankatax_ai.backend.chat.schemas import AnswerDraft, CondensedQuestion
from lankatax_ai.backend.chat.types import AskState, SearchRunner
from lankatax_ai.backend.guardrails.input_validator import InputValidator
from lankatax_ai.backend.guardrails.output_checker import OutputChecker
from lankatax_ai.backend.guardrails.types import (
    GuardrailReason,
    InputGuardResult,
    InputVerdict,
    OutputGuardResult,
    OutputVerdict,
)
from lankatax_ai.backend.memory.types import Turn
from lankatax_ai.backend.rag.prompts import get_prompt

__all__ = ["AskNodes"]

logger = logging.getLogger(__name__)


def _history_text(history: list[Turn], limit_chars: int = HISTORY_CHAR_LIMIT) -> str:
    text = "\n".join(f"{turn.role}: {turn.content}" for turn in history)
    return text[-limit_chars:]


def _timings(state: AskState, stage: str, started: float) -> dict[str, int]:
    timings = dict(state.get("timings") or {})
    timings[stage] = int((time.perf_counter() - started) * 1000)
    return timings


def _usage(state: AskState, calls: int = 0) -> dict[str, int]:
    usage = dict(state.get("usage") or {})
    usage["llm_calls"] = usage.get("llm_calls", 0) + calls
    return usage


class AskNodes:
    """Node implementations bound to one model, search runner and config."""

    def __init__(
        self,
        *,
        llm: BaseChatModel,
        search: SearchRunner,
        min_score: float,
        guard_input: bool,
        guard_output: bool,
        input_validator: InputValidator | None = None,
        output_checker: OutputChecker | None = None,
        answer_prompt_version: str | None = None,
        condense_prompt_version: str | None = None,
    ) -> None:
        self._search = search
        self._min_score = min_score
        self._guard_input = guard_input
        self._guard_output = guard_output
        self._input_validator = input_validator
        self._output_checker = output_checker
        self._answer_prompt = get_prompt("answer", version=answer_prompt_version)
        self._condense_prompt = get_prompt("condense", version=condense_prompt_version)
        self._answer_llm = llm.with_structured_output(AnswerDraft)
        self._condense_llm = llm.with_structured_output(CondensedQuestion)

    async def guard_input(self, state: AskState) -> AskState:
        started = time.perf_counter()
        question = state["question"]
        update: AskState = {
            "question": question,
            "standalone_query": question,
            "timings": _timings(state, "guard_input", started),
        }
        if not self._guard_input or self._input_validator is None:
            return update

        result: InputGuardResult = await self._input_validator.validate(
            question, history=_history_text(state.get("history") or [])
        )
        update["input_guard"] = result
        update["question"] = result.question or question
        update["standalone_query"] = result.question or question
        update["usage"] = _usage(state, calls=1)
        if result.verdict is InputVerdict.refuse:
            update["refused"] = True
            update["reason"] = result.reason
        return update

    async def condense(self, state: AskState) -> AskState:
        started = time.perf_counter()
        history = state.get("history") or []
        question = state.get("question") or state["question"]
        if not history:
            return {
                "standalone_query": question,
                "timings": _timings(state, "condense", started),
            }

        try:
            messages = self._condense_prompt.format_messages(
                history=_history_text(history),
                question=question,
            )
            raw = await self._condense_llm.ainvoke(messages)
            parsed = CondensedQuestion.model_validate(raw)
            query = parsed.question.strip() or question
        except Exception:
            logger.warning("Question condensing failed; using the raw question", exc_info=True)
            query = question

        return {
            "standalone_query": query,
            "usage": _usage(state, calls=1),
            "timings": _timings(state, "condense", started),
        }

    async def retrieve(self, state: AskState) -> AskState:
        started = time.perf_counter()
        outcome = await self._search(state.get("standalone_query") or state["question"])
        return {
            "retrieved": list(outcome.results),
            "context": outcome.context or "",
            "timings": _timings(state, "retrieve", started),
        }

    async def assess_context(self, state: AskState) -> AskState:
        started = time.perf_counter()
        retrieved = state.get("retrieved") or []
        context = (state.get("context") or "").strip()
        timings = _timings(state, "assess_context", started)

        if not retrieved or not context:
            return {
                "refused": True,
                "reason": GuardrailReason.insufficient_context,
                "timings": timings,
            }

        if max(result.score for result in retrieved) < self._min_score:
            return {
                "refused": True,
                "reason": GuardrailReason.insufficient_context,
                "timings": timings,
            }

        return {"timings": timings}

    async def answer(self, state: AskState) -> AskState:
        return await self._answer(state, strict=False)

    async def answer_strict(self, state: AskState) -> AskState:
        return await self._answer(state, strict=True)

    async def _answer(self, state: AskState, *, strict: bool) -> AskState:
        started = time.perf_counter()
        question = state.get("question") or state["question"]
        if strict:
            question = f"{question}{STRICT_SUFFIX}"

        messages = self._answer_prompt.format_messages(
            question=question,
            context=state.get("context") or "",
            history=_history_text(state.get("history") or []),
        )
        raw = await self._answer_llm.ainvoke(messages)
        draft = AnswerDraft.model_validate(raw)

        update: AskState = {
            "draft": draft,
            "answer": draft.answer.strip(),
            "attempts": state.get("attempts", 0) + (1 if strict else 0),
            "usage": _usage(state, calls=1),
            "timings": _timings(state, "answer", started),
        }
        if draft.insufficient_context:
            update["refused"] = True
            update["reason"] = GuardrailReason.insufficient_context
        return update

    async def guard_output(self, state: AskState) -> AskState:
        started = time.perf_counter()
        answer = state.get("answer") or ""
        timings = _timings(state, "guard_output", started)

        if state.get("refused"):
            return {"timings": timings}
        if not self._guard_output or self._output_checker is None:
            return {
                "output_guard": OutputGuardResult(OutputVerdict.ok, answer=answer),
                "timings": timings,
            }

        result = await self._output_checker.check(
            question=state.get("question") or "",
            answer=answer,
            context=state.get("context") or "",
            citation_count=len(state.get("retrieved") or []),
        )
        update: AskState = {
            "output_guard": result,
            "answer": result.answer or answer,
            "usage": _usage(state, calls=1),
            "timings": timings,
        }
        if result.verdict is OutputVerdict.refuse:
            update["refused"] = True
            update["reason"] = result.reason
        elif result.verdict is OutputVerdict.retry and state.get("attempts", 0) >= 1:
            update["refused"] = True
            update["reason"] = result.reason or GuardrailReason.ungrounded
        return update

    async def refuse(self, state: AskState) -> AskState:
        started = time.perf_counter()
        reason = state.get("reason")
        return {
            "answer": REFUSAL_MESSAGES.get(reason, DEFAULT_REFUSAL) if reason else DEFAULT_REFUSAL,
            "refused": True,
            "reason": reason,
            "timings": _timings(state, "refuse", started),
        }

    async def finalize(self, state: AskState) -> AskState:
        started = time.perf_counter()
        return {
            "refused": False,
            "timings": _timings(state, "finalize", started),
        }
