from __future__ import annotations

from types import SimpleNamespace

import pytest

from lankatax_ai.backend.jobs.contracts import ScrapeInput
from lankatax_ai.backend.jobs.tasks import scrape as scrape_task
from lankatax_ai.backend.scraper.crawler import (
    AsyncCrawlerErrorKind,
    AsyncCrawlerFetchResult,
    AsyncCrawlerParsedPage,
)
from lankatax_ai.backend.services.scrape import (
    CrawlBatch,
    CrawlPhase,
    SaveOutcome,
    ScrapeBatch,
)
from tests.helpers import make_scrapable_url_entry

pytestmark = pytest.mark.unit


class _Ctx:
    def __init__(self, cancelled: bool = False) -> None:
        self.is_cancelled = cancelled
        self.refreshes = 0
        self.logs: list[object] = []
        self.cancelled = False

    def refresh_timeout(self, _duration) -> None:
        self.refreshes += 1

    def log(self, line: object) -> None:
        self.logs.append(line)

    async def aio_cancel(self) -> None:
        self.cancelled = True


def _success(url: str) -> AsyncCrawlerFetchResult:
    return AsyncCrawlerFetchResult(
        url=url,
        entry=make_scrapable_url_entry(url=url),
        parsed_page=AsyncCrawlerParsedPage(
            title="title",
            text="text",
            raw_html="<p>text</p>",
            pdf_links=[],
        ),
    )


def _failure(url: str) -> AsyncCrawlerFetchResult:
    return AsyncCrawlerFetchResult(
        url=url,
        entry=make_scrapable_url_entry(url=url),
        error=AsyncCrawlerErrorKind.unknown_error,
    )


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(session_factory=None, scrape=None)


async def test_run_scrape_aggregates_batches(monkeypatch):
    batches = [
        ScrapeBatch(
            crawl=CrawlBatch(
                phase=CrawlPhase.registry,
                results=[_success("https://www.ird.gov.lk/a")],
                discovered=(make_scrapable_url_entry(url="https://www.ird.gov.lk/a.pdf"),),
            ),
            save=SaveOutcome(saved=1, skipped=2),
        )
    ]

    async def fake_batches(*args, **kwargs):
        for batch in batches:
            yield batch

    monkeypatch.setattr(scrape_task, "scrape_batches", fake_batches)
    ctx = _Ctx()

    output = await scrape_task.run_scrape(ScrapeInput(), ctx, _runtime())  # type: ignore[arg-type]

    assert output.saved == 1
    assert output.skipped == 2
    assert output.discovered == 1
    assert output.failed == 0
    assert output.partial is False
    assert ctx.refreshes == 1
    assert ctx.logs


async def test_run_scrape_reports_partial_failures(monkeypatch):
    async def fake_batches(*args, **kwargs):
        yield ScrapeBatch(
            crawl=CrawlBatch(
                phase=CrawlPhase.registry,
                results=[_failure("https://www.ird.gov.lk/bad")],
            ),
            save=SaveOutcome(saved=0, skipped=0),
        )

    monkeypatch.setattr(scrape_task, "scrape_batches", fake_batches)

    output = await scrape_task.run_scrape(ScrapeInput(), _Ctx(), _runtime())  # type: ignore[arg-type]

    assert output.failed == 1
    assert output.partial is True
    assert output.errors == ["https://www.ird.gov.lk/bad: unknown_error"]


async def test_run_scrape_cooperates_with_cancellation(monkeypatch):
    async def fake_batches(*args, **kwargs):
        yield ScrapeBatch(
            crawl=CrawlBatch(
                phase=CrawlPhase.registry, results=[_success("https://www.ird.gov.lk/a")]
            ),
            save=SaveOutcome(saved=1, skipped=0),
        )
        yield ScrapeBatch(
            crawl=CrawlBatch(
                phase=CrawlPhase.registry, results=[_success("https://www.ird.gov.lk/b")]
            ),
            save=SaveOutcome(saved=1, skipped=0),
        )

    monkeypatch.setattr(scrape_task, "scrape_batches", fake_batches)
    ctx = _Ctx(cancelled=True)

    output = await scrape_task.run_scrape(ScrapeInput(), ctx, _runtime())  # type: ignore[arg-type]

    assert output.saved == 0
    assert ctx.cancelled is True
