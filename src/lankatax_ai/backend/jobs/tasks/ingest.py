"""Ingest task execution."""

from __future__ import annotations

import time
from contextlib import aclosing
from uuid import UUID

from hatchet_sdk.context.context import Context

from lankatax_ai.backend.ingestion.pipeline import ingest_batches
from lankatax_ai.backend.ingestion.types import ChunkConfig
from lankatax_ai.backend.jobs.contracts import (
    IngestDocumentInput,
    IngestInput,
    IngestOutput,
)
from lankatax_ai.backend.jobs.deps import JobRuntime
from lankatax_ai.backend.jobs.observability import (
    job_event,
    record_job_completed,
    record_job_started,
)
from lankatax_ai.backend.jobs.policies import INGEST_HEARTBEAT
from lankatax_ai.backend.services.corpus import finalize_ingest
from lankatax_ai.backend.services.jobs.types import JobType

__all__ = ["run_ingest", "run_ingest_document"]

_MAX_ERRORS = 20


async def _run_documents(
    *,
    ctx: Context,
    runtime: JobRuntime,
    doc_ids: list[UUID],
    max_tokens: int | None,
    job_type: JobType,
) -> IngestOutput:
    started = time.perf_counter()
    record_job_started(job_type)
    docs_processed = 0
    chunks_created = 0
    tokens_used = 0
    budget_exhausted = False
    errors: list[str] = []

    async with runtime.session_factory() as session:
        repo = runtime.ingest.repository_factory(session)
        async with aclosing(
            ingest_batches(
                repo,
                doc_ids,
                ChunkConfig(),
                runtime.ingest.embeddings,
                runtime.ingest.embedding_policy,
                max_tokens=max_tokens,
            )
        ) as batches:
            async for batch in batches:
                if ctx.is_cancelled:
                    await ctx.aio_cancel()
                    break

                if batch.error:
                    if len(errors) < _MAX_ERRORS:
                        errors.append(f"{batch.document_id}: {batch.error}")
                    budget_exhausted = budget_exhausted or batch.budget_exhausted
                else:
                    docs_processed += 1
                    chunks_created += batch.chunks_created
                    tokens_used += batch.tokens_used

                ctx.refresh_timeout(INGEST_HEARTBEAT)
                job_event(
                    ctx,
                    event="ingest.document",
                    job_type=job_type,
                    document_id=str(batch.document_id),
                    chunks=batch.chunks_created,
                    tokens=batch.tokens_used,
                    error=batch.error,
                )

    await finalize_ingest(
        session_factory=runtime.session_factory,
        run_id=ctx.workflow_run_id,
    )

    duration_ms = int((time.perf_counter() - started) * 1000)
    status = "cancelled" if ctx.is_cancelled else ("partial" if errors else "succeeded")
    record_job_completed(
        job_type,
        status=status,
        duration_ms=duration_ms,
        tokens=tokens_used,
    )
    return IngestOutput(
        docs_processed=docs_processed,
        chunks_created=chunks_created,
        tokens_used=tokens_used,
        partial=bool(errors) or budget_exhausted,
        budget_exhausted=budget_exhausted,
        errors=errors,
        duration_ms=duration_ms,
    )


async def run_ingest(
    payload: IngestInput,
    ctx: Context,
    runtime: JobRuntime,
) -> IngestOutput:
    """Ingest pending or all documents, then finalize the corpus epoch."""
    async with runtime.session_factory() as session:
        repo = runtime.ingest.repository_factory(session)
        if payload.target == "all":
            doc_ids = await repo.all_document_ids()
        else:
            doc_ids = await repo.pending_document_ids()

    if payload.max_documents is not None:
        doc_ids = doc_ids[: payload.max_documents]

    return await _run_documents(
        ctx=ctx,
        runtime=runtime,
        doc_ids=doc_ids,
        max_tokens=payload.max_tokens,
        job_type=JobType.ingest_pending,
    )


async def run_ingest_document(
    payload: IngestDocumentInput,
    ctx: Context,
    runtime: JobRuntime,
) -> IngestOutput:
    """Ingest one document by id, then finalize the corpus epoch."""
    return await _run_documents(
        ctx=ctx,
        runtime=runtime,
        doc_ids=[payload.document_id],
        max_tokens=payload.max_tokens,
        job_type=JobType.ingest_document,
    )
