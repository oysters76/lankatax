"""The scrape use case: crawl the registry, discover PDFs, save documents.

Transport-agnostic. The CLI adapts progress events to Rich and asks the user
whether to save partial results; this module only crawls, deduplicates and
persists.
"""

from __future__ import annotations

from collections.abc import Callable

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
) -> CrawlOutcome:
    """Crawl the registry and, unless disabled, newly discovered PDFs.

    A ``KeyboardInterrupt`` stops the crawl and returns the results collected so
    far with ``interrupted=True``; saving partials is the caller's decision.
    ``should_stop`` is checked between batches so a job can cancel cooperatively.
    """
    entries = deps.registry.get_all()
    results: list[AsyncCrawlerFetchResult] = []

    def stop_requested() -> bool:
        return should_stop is not None and should_stop()

    def collector(phase: CrawlPhase) -> Callable[[AsyncCrawlerFetchResult], None]:
        def collect(result: AsyncCrawlerFetchResult) -> None:
            results.append(result)
            if on_progress:
                on_progress(phase, result)

        return collect

    def batch_collector(
        phase: CrawlPhase,
    ) -> Callable[[list[AsyncCrawlerFetchResult]], None]:
        def collect(batch: list[AsyncCrawlerFetchResult]) -> None:
            if on_batch:
                on_batch(phase, batch)

        return collect

    if stop_requested():
        return CrawlOutcome(
            results=[],
            succeeded=0,
            failed=0,
            discovered=0,
            cancelled=True,
        )

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
            async with session_factory() as session:
                known = await deps.store_factory(session).known_urls()
            registry_urls = {entry.url for entry in entries}
            discovered = discover_pdf_links(results, known, registry_urls)
            discovered_count = len(discovered)
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

    succeeded = sum(1 for result in results if not result.is_error)
    errors = [
        f"{result.url}: {result.error.value}" for result in results if result.error is not None
    ]
    return CrawlOutcome(
        results=results,
        succeeded=succeeded,
        failed=len(results) - succeeded,
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
            hashes = await store.hash_map()
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
    """Crawl and save in one call.

    Interrupted or cancelled crawls are returned without saving; the caller may
    inspect the partial results and call :func:`save_results` if the user agrees.
    """
    crawl_outcome = await crawl(
        options,
        session_factory=session_factory,
        deps=deps,
        on_progress=on_progress,
        on_batch=on_batch,
        should_stop=should_stop,
        batch_size=batch_size,
    )
    save_outcome: SaveOutcome | None = None
    if not crawl_outcome.interrupted and not crawl_outcome.cancelled:
        save_outcome = await save_results(
            crawl_outcome.results,
            full=options.full,
            session_factory=session_factory,
            deps=deps,
        )
    return ScrapeOutcome(crawl=crawl_outcome, save=save_outcome)


async def get_scrape_status(
    *, session_factory: SessionFactory, deps: ScrapeDeps
) -> ScrapeStatusRecord:
    """Return scrape totals and the first/last scrape timestamps."""
    async with session_factory() as session:
        return await deps.store_factory(session).scrape_status()
