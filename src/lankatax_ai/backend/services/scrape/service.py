"""The scrape use case: crawl the registry, discover PDFs, save documents.

Transport-agnostic. The CLI adapts progress events to Rich and asks the user
whether to save partial results; this module only crawls, deduplicates and
persists.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable

from lankatax_ai.backend.documents import ScrapeStatusRecord
from lankatax_ai.backend.scraper.crawler import AsyncCrawlerFetchResult
from lankatax_ai.backend.scraper.pdf_discovery import discover_pdf_links
from lankatax_ai.backend.scraper.registry import ScrapableURL
from lankatax_ai.backend.scraper.transform import to_document
from lankatax_ai.backend.services.protocols import SessionFactory
from lankatax_ai.backend.services.scrape.types import (
    BatchCallback,
    CrawlOutcome,
    CrawlPhase,
    ProgressCallback,
    SaveOutcome,
    ScrapeDeps,
    ScrapeOptions,
    ScrapeOutcome,
    StopCheck,
)


def list_urls(deps: ScrapeDeps) -> list[ScrapableURL]:
    """Return the configured registry entries."""
    return deps.registry.get_all()


async def crawl(
    options: ScrapeOptions,
    *,
    session_factory: SessionFactory,
    deps: ScrapeDeps,
    on_progress: ProgressCallback | None = None,
    on_batch: BatchCallback | None = None,
    should_stop: StopCheck | None = None,
    batch_size: int = 25,
    retain_results: bool = True,
) -> CrawlOutcome:
    """Crawl the registry and, unless disabled, newly discovered PDFs.

    A ``KeyboardInterrupt`` stops the crawl and returns the results collected so
    far with ``interrupted=True``; saving partials is the caller's decision.
    ``should_stop`` is checked between batches so a job can cancel cooperatively.
    Set ``retain_results=False`` when a batch callback persists each batch; the
    outcome then carries counts but no result payloads.
    """
    entries = deps.registry.get_all()
    results: list[AsyncCrawlerFetchResult] = []
    errors: list[str] = []
    discovered: list[ScrapableURL] = []
    registry_urls = {entry.url for entry in entries}
    succeeded = 0
    failed = 0

    def stop_requested() -> bool:
        return should_stop is not None and should_stop()

    if stop_requested():
        return CrawlOutcome(
            results=[],
            succeeded=0,
            failed=0,
            discovered=0,
            cancelled=True,
        )

    known_for_discovery: set[str] = set()
    if options.discover_pdfs:
        async with session_factory() as session:
            known_for_discovery = await deps.store_factory(session).known_urls()

    def collector(phase: CrawlPhase) -> Callable[[AsyncCrawlerFetchResult], None]:
        def collect(result: AsyncCrawlerFetchResult) -> None:
            nonlocal succeeded, failed
            if result.is_error:
                failed += 1
                errors.append(f"{result.url}: {result.error.value if result.error else 'unknown'}")
            else:
                succeeded += 1
            if retain_results:
                results.append(result)
            if on_progress:
                on_progress(phase, result)

        return collect

    def batch_collector(
        phase: CrawlPhase,
    ) -> Callable[[list[AsyncCrawlerFetchResult]], Awaitable[None]]:
        async def collect(batch: list[AsyncCrawlerFetchResult]) -> None:
            if phase is CrawlPhase.registry and options.discover_pdfs:
                for pdf in discover_pdf_links(batch, known_for_discovery, registry_urls):
                    discovered.append(pdf)
                    known_for_discovery.add(pdf.url)
            if on_batch:
                pending = on_batch(phase, batch)
                if inspect.isawaitable(pending):
                    await pending

        return collect

    interrupted = False
    cancelled = False
    discovered_count = 0

    try:
        async with deps.crawler_factory() as crawler:
            await crawler.crawl_all(
                entries,
                on_result=collector(CrawlPhase.registry),
                on_batch=batch_collector(CrawlPhase.registry),
                should_stop=should_stop,
                batch_size=batch_size,
            )

        cancelled = stop_requested()

        if options.discover_pdfs and not cancelled:
            if discovered and not stop_requested():
                async with deps.crawler_factory() as crawler:
                    await crawler.crawl_all(
                        discovered,
                        on_result=collector(CrawlPhase.discovered),
                        on_batch=batch_collector(CrawlPhase.discovered),
                        should_stop=should_stop,
                        batch_size=batch_size,
                    )
                cancelled = stop_requested()
            elif discovered:
                cancelled = True
    except KeyboardInterrupt:
        interrupted = True

    discovered_count = len(discovered)
    return CrawlOutcome(
        results=results,
        succeeded=succeeded,
        failed=failed,
        discovered=discovered_count,
        errors=errors,
        interrupted=interrupted,
        cancelled=cancelled,
    )


async def save_results(
    results: list[AsyncCrawlerFetchResult],
    *,
    full: bool,
    session_factory: SessionFactory,
    deps: ScrapeDeps,
) -> SaveOutcome:
    """Persist fetched documents, skipping unchanged URLs unless ``full``."""
    records = [doc for result in results if (doc := to_document(result)) is not None]
    if not records:
        return SaveOutcome(saved=0, skipped=0)

    async with session_factory() as session:
        store = deps.store_factory(session)
        if full:
            to_save = records
        else:
            hashes = await store.hash_map([doc.url for doc in records])
            to_save = [
                doc
                for doc in records
                if doc.url not in hashes or doc.content_hash != hashes[doc.url]
            ]

        skipped = len(records) - len(to_save)
        if not to_save:
            return SaveOutcome(saved=0, skipped=skipped)

        await store.save(to_save)
        await store.commit()
        return SaveOutcome(saved=len(to_save), skipped=skipped)


async def scrape(
    options: ScrapeOptions,
    *,
    session_factory: SessionFactory,
    deps: ScrapeDeps,
    on_progress: ProgressCallback | None = None,
    on_batch: BatchCallback | None = None,
    should_stop: StopCheck | None = None,
    batch_size: int = 25,
) -> ScrapeOutcome:
    """Crawl and persist in batches.

    Each completed batch is saved before the next batch starts, so memory stays
    bounded by ``batch_size``. A cancelled or interrupted crawl keeps the
    batches that already committed; the outcome still exposes that it was
    incomplete. The caller can inspect counts but no longer needs the result
    payloads.
    """
    saved = 0
    skipped = 0
    persisted_batches = 0

    async def persist_batch(phase: CrawlPhase, batch: list[AsyncCrawlerFetchResult]) -> None:
        nonlocal saved, skipped, persisted_batches
        outcome = await save_results(
            batch,
            full=options.full,
            session_factory=session_factory,
            deps=deps,
        )
        saved += outcome.saved
        skipped += outcome.skipped
        persisted_batches += 1
        if on_batch:
            pending = on_batch(phase, batch)
            if inspect.isawaitable(pending):
                await pending

    crawl_outcome = await crawl(
        options,
        session_factory=session_factory,
        deps=deps,
        on_progress=on_progress,
        on_batch=persist_batch,
        should_stop=should_stop,
        batch_size=batch_size,
        retain_results=False,
    )
    save_outcome: SaveOutcome | None = None
    if persisted_batches or (not crawl_outcome.interrupted and not crawl_outcome.cancelled):
        save_outcome = SaveOutcome(saved=saved, skipped=skipped)
    return ScrapeOutcome(crawl=crawl_outcome, save=save_outcome)


async def get_scrape_status(
    *, session_factory: SessionFactory, deps: ScrapeDeps
) -> ScrapeStatusRecord:
    """Return scrape totals and the first/last scrape timestamps."""
    async with session_factory() as session:
        return await deps.store_factory(session).scrape_status()
