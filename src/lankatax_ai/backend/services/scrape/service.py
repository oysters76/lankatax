"""The scrape use case: crawl the registry, discover PDFs, save documents.

Transport-agnostic. Callers either consume crawl/scrape batches or collect a
single outcome.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from lankatax_ai.backend.documents import ScrapeStatusRecord
from lankatax_ai.backend.scraper.crawler import AsyncCrawlerFetchResult
from lankatax_ai.backend.scraper.pdf_discovery import discover_pdf_links
from lankatax_ai.backend.scraper.registry import ScrapableURL
from lankatax_ai.backend.scraper.transform import to_document
from lankatax_ai.backend.services.protocols import SessionFactory
from lankatax_ai.backend.services.scrape.types import (
    CrawlBatch,
    CrawlOutcome,
    CrawlPhase,
    CrawlSummary,
    SaveOutcome,
    ScrapeBatch,
    ScrapeDeps,
    ScrapeOptions,
    ScrapeOutcome,
)


def list_urls(deps: ScrapeDeps) -> list[ScrapableURL]:
    """Return the configured registry entries."""
    return deps.registry.get_all()


async def crawl_batches(
    options: ScrapeOptions,
    *,
    session_factory: SessionFactory,
    deps: ScrapeDeps,
    batch_size: int = 25,
) -> AsyncIterator[CrawlBatch]:
    """Yield completed registry and discovered-PDF batches.

    Consumption controls backpressure and cancellation: stop iterating and the
    crawler context closes. Discovery data travels with the registry batch that
    produced it.
    """
    entries = deps.registry.get_all()
    registry_urls = {entry.url for entry in entries}
    known_for_discovery: set[str] = set()
    if options.discover_pdfs:
        async with session_factory() as session:
            known_for_discovery = await deps.store_factory(session).known_urls()

    discovered: list[ScrapableURL] = []

    async with deps.crawler_factory() as crawler:
        async for batch in crawler.crawl_batches(entries, batch_size=batch_size):
            new_pdfs: list[ScrapableURL] = []
            if options.discover_pdfs:
                new_pdfs = discover_pdf_links(batch, known_for_discovery, registry_urls)
                for pdf in new_pdfs:
                    discovered.append(pdf)
                    known_for_discovery.add(pdf.url)
            yield CrawlBatch(
                phase=CrawlPhase.registry,
                results=batch,
                discovered=tuple(new_pdfs),
            )

    if options.discover_pdfs and discovered:
        async with deps.crawler_factory() as crawler:
            async for batch in crawler.crawl_batches(discovered, batch_size=batch_size):
                yield CrawlBatch(phase=CrawlPhase.discovered, results=batch)


def _accumulate(
    summary: CrawlSummary, batch: CrawlBatch, results: list[AsyncCrawlerFetchResult] | None = None
) -> None:
    summary.discovered += len(batch.discovered)
    for result in batch.results:
        if result.is_error:
            summary.failed += 1
            summary.errors.append(
                f"{result.url}: {result.error.value if result.error else 'unknown'}"
            )
        else:
            summary.succeeded += 1
        if results is not None:
            results.append(result)


async def crawl(
    options: ScrapeOptions,
    *,
    session_factory: SessionFactory,
    deps: ScrapeDeps,
    batch_size: int = 25,
) -> CrawlOutcome:
    """Collect every crawl batch into one outcome.

    Streaming callers use :func:`crawl_batches` instead.
    """
    summary = CrawlSummary(succeeded=0, failed=0, discovered=0)
    results: list[AsyncCrawlerFetchResult] = []
    interrupted = False

    try:
        async for batch in crawl_batches(
            options,
            session_factory=session_factory,
            deps=deps,
            batch_size=batch_size,
        ):
            _accumulate(summary, batch, results)
    except KeyboardInterrupt:
        interrupted = True

    return CrawlOutcome(
        results=results,
        succeeded=summary.succeeded,
        failed=summary.failed,
        discovered=summary.discovered,
        errors=summary.errors,
        interrupted=interrupted,
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


async def scrape_batches(
    options: ScrapeOptions,
    *,
    session_factory: SessionFactory,
    deps: ScrapeDeps,
    batch_size: int = 25,
) -> AsyncIterator[ScrapeBatch]:
    """Yield crawl batches after each one has been persisted.

    Stop iterating to stop the crawl. Batches already yielded stay committed.
    """
    async for batch in crawl_batches(
        options,
        session_factory=session_factory,
        deps=deps,
        batch_size=batch_size,
    ):
        saved = await save_results(
            batch.results,
            full=options.full,
            session_factory=session_factory,
            deps=deps,
        )
        yield ScrapeBatch(crawl=batch, save=saved)


async def scrape(
    options: ScrapeOptions,
    *,
    session_factory: SessionFactory,
    deps: ScrapeDeps,
    batch_size: int = 25,
) -> ScrapeOutcome:
    """Persist every batch and collect the run summary.

    Streaming callers use :func:`scrape_batches` instead.
    """
    summary = CrawlSummary(succeeded=0, failed=0, discovered=0)
    saved = 0
    skipped = 0
    persisted_batches = 0
    interrupted = False

    try:
        async for batch in scrape_batches(
            options,
            session_factory=session_factory,
            deps=deps,
            batch_size=batch_size,
        ):
            _accumulate(summary, batch.crawl)
            saved += batch.save.saved
            skipped += batch.save.skipped
            persisted_batches += 1
    except KeyboardInterrupt:
        interrupted = True

    summary.interrupted = interrupted
    save_outcome: SaveOutcome | None = None
    if persisted_batches or not interrupted:
        save_outcome = SaveOutcome(saved=saved, skipped=skipped)
    return ScrapeOutcome(crawl=summary, save=save_outcome)


async def get_scrape_status(
    *, session_factory: SessionFactory, deps: ScrapeDeps
) -> ScrapeStatusRecord:
    """Return scrape totals and the first/last scrape timestamps."""
    async with session_factory() as session:
        return await deps.store_factory(session).scrape_status()
