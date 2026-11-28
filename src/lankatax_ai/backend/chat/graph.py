"""LangGraph wiring for the ask flow.

The graph is stateless: conversation history arrives in the initial state and
the service persists the result. That keeps LangGraph out of storage concerns.
"""

from __future__ import annotations

import dataclasses

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from lankatax_ai.backend.chat.nodes import AskNodes, SearchRunner
from lankatax_ai.backend.chat.state import AskState
from lankatax_ai.backend.guardrails.input_validator import InputValidator
from lankatax_ai.backend.guardrails.output_checker import OutputChecker
from lankatax_ai.backend.guardrails.types import OutputVerdict

__all__ = ["GraphConfig", "GraphDeps", "build_ask_graph"]


@dataclasses.dataclass(frozen=True, slots=True)
class GraphConfig:
    """Guardrail and prompt choices for one graph build."""

    min_score: float = 0.2
    guard_input: bool = True
    guard_output: bool = True
    answer_prompt_version: str | None = None
    condense_prompt_version: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class GraphDeps:
    """Everything the graph needs; built by the ask use case wiring."""

    llm: BaseChatModel
    search: SearchRunner
    config: GraphConfig = dataclasses.field(default_factory=GraphConfig)
    input_validator: InputValidator | None = None
    output_checker: OutputChecker | None = None


def _after_input(state: AskState) -> str:
    return "refuse" if state.get("refused") else "condense"


def _after_assess(state: AskState) -> str:
    return "refuse" if state.get("refused") else "answer"


def _after_output(state: AskState) -> str:
    if state.get("refused"):
        return "refuse"
    guard = state.get("output_guard")
    if guard is not None and guard.verdict is OutputVerdict.retry:
        return "answer_strict"
    return "finalize"


def build_ask_graph(
    deps: GraphDeps,
) -> CompiledStateGraph[AskState, None, AskState, AskState]:
    """Compile the ask graph: guard → condense → retrieve → answer → guard."""
    nodes = AskNodes(
        llm=deps.llm,
        search=deps.search,
        min_score=deps.config.min_score,
        guard_input=deps.config.guard_input,
        guard_output=deps.config.guard_output,
        input_validator=deps.input_validator,
        output_checker=deps.output_checker,
        answer_prompt_version=deps.config.answer_prompt_version,
        condense_prompt_version=deps.config.condense_prompt_version,
    )

    graph = StateGraph(AskState)
    graph.add_node("guard_input", nodes.guard_input)
    graph.add_node("condense", nodes.condense)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("assess_context", nodes.assess_context)
    graph.add_node("answer", nodes.answer)
    graph.add_node("answer_strict", nodes.answer_strict)
    graph.add_node("guard_output", nodes.guard_output)
    graph.add_node("refuse", nodes.refuse)
    graph.add_node("finalize", nodes.finalize)

    graph.set_entry_point("guard_input")
    graph.add_conditional_edges(
        "guard_input",
        _after_input,
        {"refuse": "refuse", "condense": "condense"},
    )
    graph.add_edge("condense", "retrieve")
    graph.add_edge("retrieve", "assess_context")
    graph.add_conditional_edges(
        "assess_context",
        _after_assess,
        {"refuse": "refuse", "answer": "answer"},
    )
    graph.add_edge("answer", "guard_output")
    graph.add_conditional_edges(
        "guard_output",
        _after_output,
        {"refuse": "refuse", "answer_strict": "answer_strict", "finalize": "finalize"},
    )
    graph.add_edge("answer_strict", "guard_output")
    graph.add_edge("refuse", END)
    graph.add_edge("finalize", END)

    return graph.compile()
