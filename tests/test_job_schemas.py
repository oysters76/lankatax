from __future__ import annotations

from datetime import UTC, datetime

import pytest

from lankatax_ai.backend.api.schemas import (
    JobDetailsSchema,
    JobReceiptSchema,
    JobRequestSchema,
    JobTaskSchema,
)
from lankatax_ai.backend.services.jobs import JobStatus, JobType

pytestmark = pytest.mark.unit

NOW = datetime(2026, 11, 28, tzinfo=UTC)


def test_job_request_schema_parses_type_and_json_params():
    payload = JobRequestSchema(
        type="pipeline.refresh",
        params={"full": True, "max_tokens": 1000},
        idempotency_key="key-1",
    )

    assert payload.type is JobType.pipeline_refresh
    assert payload.params == {"full": True, "max_tokens": 1000}
    assert payload.idempotency_key == "key-1"


def test_job_details_schema_contains_tasks():
    task = JobTaskSchema(
        id="task-1",
        name="scrape-registry",
        status=JobStatus.succeeded,
        output={"saved": 1},
    )
    details = JobDetailsSchema(
        id="run-1",
        type=JobType.pipeline_refresh,
        status=JobStatus.succeeded,
        created_at=NOW,
        input={"full": True},
        output={"saved": 1},
        tasks=[task],
    )

    assert details.tasks[0].name == "scrape-registry"
    assert details.output == {"saved": 1}


def test_job_receipt_schema_marks_reuse():
    receipt = JobReceiptSchema(run_id="run-1", reused=True)

    assert receipt.reused is True
