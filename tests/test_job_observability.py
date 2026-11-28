from __future__ import annotations

import pytest

from lankatax_ai.backend.jobs.observability import job_event, job_trace_metadata
from lankatax_ai.backend.services.jobs import JobType

pytestmark = pytest.mark.unit


class _Ctx:
    workflow_run_id = "run-1"
    task_run_id = "task-1"
    retry_count = 2

    def __init__(self) -> None:
        self.logs: list[object] = []

    def log(self, line: object) -> None:
        self.logs.append(line)


def test_job_event_includes_correlation_fields():
    ctx = _Ctx()

    job_event(
        ctx,  # type: ignore[arg-type]
        event="scrape.batch",
        job_type=JobType.scrape_registry,
        saved=2,
    )

    assert ctx.logs == [
        {
            "event": "scrape.batch",
            "job_type": "scrape.registry",
            "run_id": "run-1",
            "task_run_id": "task-1",
            "attempt": 2,
            "saved": 2,
        }
    ]


def test_job_trace_metadata():
    assert job_trace_metadata(
        run_id="run-1",
        job_type=JobType.pipeline_refresh,
        attempt=1,
        corpus_epoch="4",
    ) == {
        "hatchet_run_id": "run-1",
        "job_type": "pipeline.refresh",
        "attempt": 1,
        "corpus_epoch": "4",
    }
