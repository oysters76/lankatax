"""Prometheus metrics for the ask use case."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

from lankatax_ai.backend.services.ask.types import AskOutcome

__all__ = ["record_ask"]

ASK_TOTAL = Counter(
    "lankatax_ask_total",
    "Ask requests by outcome",
    ["status", "reason"],
)
ASK_STAGE = Histogram(
    "lankatax_ask_duration_seconds",
    "Ask stage duration in seconds",
    ["stage"],
)


def record_ask(outcome: AskOutcome) -> None:
    status = "refused" if outcome.refused else "answered"
    reason = outcome.reason.value if outcome.reason is not None else "none"
    ASK_TOTAL.labels(status=status, reason=reason).inc()
    for stage, milliseconds in outcome.timings.items():
        ASK_STAGE.labels(stage=stage).observe(milliseconds / 1000)
