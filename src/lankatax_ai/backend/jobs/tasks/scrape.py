"""Scrape task execution."""

from __future__ import annotations

import time
from contextlib import aclosing

from hatchet_sdk.context.context import Context

from lankatax_ai.backend.jobs.contracts import ScrapeInput, ScrapeOutput
from lankatax_ai.backend.jobs.deps import JobRuntime
from lankatax_ai.backend.jobs.observability import (
    job_event,
    record_job_completed,
    record_job_started,
)
from lankatax_ai.backend.jobs.policies import SCRAPE_HEARTBEAT
from lankatax_ai.backend.services.jobs.types import JobType
from lankatax_ai.backend.services.scrape import ScrapeOptions, scrape_batches

__all__ = ["run_scrape"]

_MAX_ERRORS = 20


async def run_scrape(payload: ScrapeInput, ctx: Context, runtime: JobRuntime) -> ScrapeOutput:
    """Consume scrape batches, stopping and reporting partial progress."""
    started = time.perf_counter()
    record_job_started(JobType.scrape_registry)
    saved = 0
    skipped = 0
    failed = 0
    discovered = 0
    errors: list[str] = []

    options = ScrapeOptions(full=payload.full, discover_pdfs=payload.discover_pdfs)
    async with aclosing(
        scrape_batches(
            options,
            session_factory=runtime.session_factory,
            deps=runtime.scrape,
            batch_size=payload.batch_size,
        )
    ) as batches:
        async for batch in batches:
            if ctx.is_cancelled:
                await ctx.aio_cancel()
                break

            saved += batch.save.saved
            skipped += batch.save.skipped
            discovered += len(batch.crawl.discovered)
            for result in batch.crawl.results:
                if result.is_error:
                    failed += 1
                    if len(errors) < _MAX_ERRORS:
                        errors.append(
                            f"{result.url}: {result.error.value if result.error else 'unknown'}"
                        )

            ctx.refresh_timeout(SCRAPE_HEARTBEAT)
            job_event(
                ctx,
                event="scrape.batch",
                job_type=JobType.scrape_registry,
                phase=batch.crawl.phase.value,
                saved=batch.save.saved,
                skipped=batch.save.skipped,
                failed=failed,
                discovered=discovered,
            )

    duration_ms = int((time.perf_counter() - started) * 1000)
    status = "cancelled" if ctx.is_cancelled else ("partial" if failed else "succeeded")
    record_job_completed(
        JobType.scrape_registry,
        status=status,
        duration_ms=duration_ms,
    )
    return ScrapeOutput(
        saved=saved,
        skipped=skipped,
        failed=failed,
        discovered=discovered,
        partial=failed > 0,
        errors=errors,
        duration_ms=duration_ms,
    )
