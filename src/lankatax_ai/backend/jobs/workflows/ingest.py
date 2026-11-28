"""Hatchet workflow registration for document ingestion."""

from __future__ import annotations

from hatchet_sdk import Hatchet
from hatchet_sdk.context.context import Context
from hatchet_sdk.runnables.workflow import Workflow

from lankatax_ai.backend.jobs.contracts import (
    IngestDocumentInput,
    IngestInput,
    IngestOutput,
)
from lankatax_ai.backend.jobs.deps import JobRuntime
from lankatax_ai.backend.jobs.policies import (
    INGEST_EXECUTION_TIMEOUT,
    INGEST_RETRIES,
    SCHEDULE_TIMEOUT,
)
from lankatax_ai.backend.jobs.tasks.ingest import run_ingest, run_ingest_document
from lankatax_ai.backend.services.jobs.types import JobType

__all__ = ["build_ingest_document_workflow", "build_ingest_pending_workflow"]


def build_ingest_pending_workflow(
    hatchet: Hatchet,
    runtime: JobRuntime,
) -> Workflow[IngestInput]:
    workflow = hatchet.workflow(
        name=JobType.ingest_pending.value,
        input_validator=IngestInput,
    )

    @workflow.task(
        name="ingest-pending",
        retries=INGEST_RETRIES,
        execution_timeout=INGEST_EXECUTION_TIMEOUT,
        schedule_timeout=SCHEDULE_TIMEOUT,
        backoff_factor=2.0,
        backoff_max_seconds=10,
    )
    async def ingest_pending(payload: IngestInput, ctx: Context) -> IngestOutput:
        return await run_ingest(payload, ctx, runtime)

    return workflow


def build_ingest_document_workflow(
    hatchet: Hatchet,
    runtime: JobRuntime,
) -> Workflow[IngestDocumentInput]:
    workflow = hatchet.workflow(
        name=JobType.ingest_document.value,
        input_validator=IngestDocumentInput,
    )

    @workflow.task(
        name="ingest-document",
        retries=INGEST_RETRIES,
        execution_timeout=INGEST_EXECUTION_TIMEOUT,
        schedule_timeout=SCHEDULE_TIMEOUT,
        backoff_factor=2.0,
        backoff_max_seconds=10,
    )
    async def ingest_document(payload: IngestDocumentInput, ctx: Context) -> IngestOutput:
        return await run_ingest_document(payload, ctx, runtime)

    return workflow
