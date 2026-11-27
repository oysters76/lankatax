from __future__ import annotations

import inspect
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Self

import pytest
from sqlalchemy import func, select

from lankatax_ai.backend.db.models import Document
from lankatax_ai.backend.documents import DocumentStore, ScrapeStatusRecord
from lankatax_ai.backend.scraper.content_hasher import compute_hash
from lankatax_ai.backend.scraper.crawler import (
    AsyncCrawlerErrorKind,
    AsyncCrawlerFetchResult,
    AsyncCrawlerParsedPage,
)
from lankatax_ai.backend.scraper.registry import ScrapableURL, URLRegistry
from lankatax_ai.backend.services.scrape import (
    CrawlPhase,
    ScrapeDeps,
    ScrapeOptions,
    build_scrape_deps,
    crawl,
    get_scrape_status,
    list_urls,
    save_results,
    scrape,
)
from lankatax_ai.backend.types import DocumentKind
from tests.helpers import make_scrapable_url_entry

pytestmark = pytest.mark.unit


class _StubSession:
    def __init__(self) -> None:
        self.closed = False


@asynccontextmanager
async def _session_factory(session: _StubSession):
    try:
        yield session
    finally:
        session.closed = True


class _StubRegistry:
    def __init__(self, entries: list[ScrapableURL]) -> None:
        self._entries = entries
        self.warnings: list[str] = []

    def get_all(self) -> list[ScrapableURL]:
        return list(self._entries)


class _StubCrawler:
    def __init__(
        self,
        *,
        pdf_links: dict[str, list[str]] | None = None,
        fail_urls: set[str] | None = None,
        interrupt_after: int | None = None,
    ) -> None:
        self._pdf_links = pdf_links or {}
        self._fail_urls = fail_urls or set()
        self._interrupt_after = interrupt_after
        self.calls: list[list[str]] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def _result(self, entry: ScrapableURL) -> AsyncCrawlerFetchResult:
        if entry.url in self._fail_urls:
            return AsyncCrawlerFetchResult(
                url=entry.url,
                entry=entry,
                error=AsyncCrawlerErrorKind.unknown_error,
            )
        return AsyncCrawlerFetchResult(
            url=entry.url,
            entry=entry,
            parsed_page=AsyncCrawlerParsedPage(
                title=entry.label,
                text=f"text for {entry.url}",
                raw_html=f"<p>text for {entry.url}</p>",
                pdf_links=self._pdf_links.get(entry.url, []),
            ),
        )

    async def crawl_all(
        self,
        entries: list[ScrapableURL],
        max_concurrent: int = 5,
        on_result=None,
        on_batch=None,
        should_stop=None,
        batch_size: int = 25,
    ) -> list[AsyncCrawlerFetchResult]:
        self.calls.append([entry.url for entry in entries])
        results: list[AsyncCrawlerFetchResult] = []
        for start in range(0, len(entries), batch_size):
            if should_stop is not None and should_stop():
                break
            batch: list[AsyncCrawlerFetchResult] = []
            for index in range(start, min(start + batch_size, len(entries))):
                if self._interrupt_after is not None and index >= self._interrupt_after:
                    raise KeyboardInterrupt
                result = self._result(entries[index])
                results.append(result)
                batch.append(result)
                if on_result:
                    on_result(result)
            if on_batch:
                pending = on_batch(batch)
                if inspect.isawaitable(pending):
                    await pending
        return results


class _StubStore:
    def __init__(
        self,
        *,
        known: set[str] | None = None,
        hashes: dict[str, str] | None = None,
    ) -> None:
        self._known = known or set()
        self._hashes = hashes or {}
        self.saved: list[Document] = []
        self.commits = 0
        now = datetime.now(UTC)
        self._status = ScrapeStatusRecord(total=3, recent=1, first_scrape=now, last_scrape=now)

    async def known_urls(self) -> set[str]:
        return set(self._known)

    async def hash_map(self, urls: list[str] | None = None) -> dict[str, str]:
        if urls is None:
            return dict(self._hashes)
        return {url: self._hashes[url] for url in urls if url in self._hashes}

    async def save(self, documents: list[Document]) -> None:
        self.saved.extend(documents)

    async def scrape_status(self) -> ScrapeStatusRecord:
        return self._status

    async def commit(self) -> None:
        self.commits += 1


def _deps(registry: _StubRegistry, crawler: _StubCrawler, store: _StubStore) -> ScrapeDeps:
    return ScrapeDeps(
        registry=registry,
        crawler_factory=lambda: crawler,
        store_factory=lambda _session: store,
    )


def _entry(url: str, *, kind: DocumentKind = DocumentKind.page) -> ScrapableURL:
    return make_scrapable_url_entry(url=url, label=url.rsplit("/", 1)[-1], kind=kind)


async def _run_crawl(
    registry: _StubRegistry,
    crawler: _StubCrawler,
    store: _StubStore,
    *,
    options: ScrapeOptions | None = None,
    on_progress=None,
    on_batch=None,
    should_stop=None,
    batch_size: int = 25,
):
    return await crawl(
        options or ScrapeOptions(),
        session_factory=lambda: _session_factory(_StubSession()),
        deps=_deps(registry, crawler, store),
        on_progress=on_progress,
        on_batch=on_batch,
        should_stop=should_stop,
        batch_size=batch_size,
    )


def test_list_urls_returns_registry_entries():
    entries = [_entry("https://www.ird.gov.lk/a"), _entry("https://www.ird.gov.lk/b")]

    assert list_urls(_deps(_StubRegistry(entries), _StubCrawler(), _StubStore())) == entries


async def test_crawl_counts_results_and_reports_errors():
    entries = [_entry("https://www.ird.gov.lk/a"), _entry("https://www.ird.gov.lk/b")]
    crawler = _StubCrawler(fail_urls={"https://www.ird.gov.lk/b"})

    outcome = await _run_crawl(_StubRegistry(entries), crawler, _StubStore())

    assert outcome.succeeded == 1
    assert outcome.failed == 1
    assert outcome.discovered == 0
    assert outcome.errors == ["https://www.ird.gov.lk/b: unknown_error"]
    assert outcome.interrupted is False


async def test_crawl_emits_phase_tagged_progress():
    entries = [_entry("https://www.ird.gov.lk/a")]
    events: list[tuple] = []

    await _run_crawl(
        _StubRegistry(entries),
        _StubCrawler(),
        _StubStore(),
        on_progress=lambda phase, result: events.append((phase, result.url)),
    )

    assert events == [(CrawlPhase.registry, "https://www.ird.gov.lk/a")]


async def test_crawl_reports_batches_and_stops_when_requested():
    entries = [
        _entry("https://www.ird.gov.lk/a"),
        _entry("https://www.ird.gov.lk/b"),
        _entry("https://www.ird.gov.lk/c"),
        _entry("https://www.ird.gov.lk/d"),
    ]
    crawler = _StubCrawler()
    batches: list[tuple[CrawlPhase, list[str]]] = []
    state = {"stop": False}

    def on_batch(phase, batch):
        batches.append((phase, [result.url for result in batch]))
        if len(batches) == 1:
            state["stop"] = True

    outcome = await _run_crawl(
        _StubRegistry(entries),
        crawler,
        _StubStore(),
        on_batch=on_batch,
        should_stop=lambda: state["stop"],
        batch_size=2,
    )

    assert outcome.cancelled is True
    assert [result.url for result in outcome.results] == [
        "https://www.ird.gov.lk/a",
        "https://www.ird.gov.lk/b",
    ]
    assert batches == [
        (
            CrawlPhase.registry,
            ["https://www.ird.gov.lk/a", "https://www.ird.gov.lk/b"],
        )
    ]
    assert crawler.calls == [[entry.url for entry in entries]]


async def test_crawl_discovers_and_crawls_new_pdfs():
    page = _entry("https://www.ird.gov.lk/page")
    known_pdf = "https://www.ird.gov.lk/known.pdf"
    new_pdf = "https://www.ird.gov.lk/new.pdf"
    crawler = _StubCrawler(pdf_links={page.url: [new_pdf, known_pdf, page.url]})

    outcome = await _run_crawl(
        _StubRegistry([page]),
        crawler,
        _StubStore(known={known_pdf}),
    )

    assert outcome.discovered == 1
    assert crawler.calls == [[page.url], [new_pdf]]
    assert outcome.succeeded == 2


async def test_crawl_can_skip_pdf_discovery():
    page = _entry("https://www.ird.gov.lk/page")
    crawler = _StubCrawler(pdf_links={page.url: ["https://www.ird.gov.lk/new.pdf"]})

    outcome = await _run_crawl(
        _StubRegistry([page]),
        crawler,
        _StubStore(),
        options=ScrapeOptions(discover_pdfs=False),
    )

    assert outcome.discovered == 0
    assert crawler.calls == [[page.url]]


async def test_crawl_returns_partial_results_on_interrupt():
    entries = [
        _entry("https://www.ird.gov.lk/a"),
        _entry("https://www.ird.gov.lk/b"),
    ]
    crawler = _StubCrawler(interrupt_after=1)

    outcome = await _run_crawl(_StubRegistry(entries), crawler, _StubStore())

    assert outcome.interrupted is True
    assert [result.url for result in outcome.results] == ["https://www.ird.gov.lk/a"]


async def test_full_save_writes_every_record():
    results = [
        AsyncCrawlerFetchResult(
            url="https://www.ird.gov.lk/a",
            entry=_entry("https://www.ird.gov.lk/a"),
            parsed_page=AsyncCrawlerParsedPage(
                title="a", text="text a", raw_html="<p>text a</p>", pdf_links=[]
            ),
        )
    ]
    store = _StubStore()

    outcome = await save_results(
        results,
        full=True,
        session_factory=lambda: _session_factory(_StubSession()),
        deps=_deps(_StubRegistry([]), _StubCrawler(), store),
    )

    assert outcome.saved == 1
    assert outcome.skipped == 0
    assert store.commits == 1


async def test_incremental_save_skips_unchanged_urls():
    unchanged = "https://www.ird.gov.lk/same"
    changed = "https://www.ird.gov.lk/changed"
    results = [
        AsyncCrawlerFetchResult(
            url=unchanged,
            entry=_entry(unchanged),
            parsed_page=AsyncCrawlerParsedPage(
                title="same",
                text=f"text for {unchanged}",
                raw_html="",
                pdf_links=[],
            ),
        ),
        AsyncCrawlerFetchResult(
            url=changed,
            entry=_entry(changed),
            parsed_page=AsyncCrawlerParsedPage(
                title="changed",
                text=f"text for {changed}",
                raw_html="",
                pdf_links=[],
            ),
        ),
    ]
    store = _StubStore(hashes={unchanged: compute_hash(f"text for {unchanged}")})

    outcome = await save_results(
        results,
        full=False,
        session_factory=lambda: _session_factory(_StubSession()),
        deps=_deps(_StubRegistry([]), _StubCrawler(), store),
    )

    assert outcome.saved == 1
    assert outcome.skipped == 1
    assert [doc.url for doc in store.saved] == [changed]


async def test_scrape_saves_when_crawl_completes():
    page = _entry("https://www.ird.gov.lk/a")
    store = _StubStore()

    outcome = await scrape(
        ScrapeOptions(),
        session_factory=lambda: _session_factory(_StubSession()),
        deps=_deps(_StubRegistry([page]), _StubCrawler(), store),
    )

    assert outcome.interrupted is False
    assert outcome.save is not None
    assert outcome.save.saved == 1


async def test_scrape_persists_each_batch_without_retaining_results():
    entries = [
        _entry("https://www.ird.gov.lk/a"),
        _entry("https://www.ird.gov.lk/b"),
        _entry("https://www.ird.gov.lk/c"),
    ]
    store = _StubStore()

    outcome = await scrape(
        ScrapeOptions(),
        session_factory=lambda: _session_factory(_StubSession()),
        deps=_deps(_StubRegistry(entries), _StubCrawler(), store),
        batch_size=1,
    )

    assert outcome.save is not None
    assert outcome.save.saved == 3
    assert len(store.saved) == 3
    assert outcome.crawl.results == []


async def test_scrape_does_not_save_interrupted_crawl():
    entries = [
        _entry("https://www.ird.gov.lk/a"),
        _entry("https://www.ird.gov.lk/b"),
    ]
    store = _StubStore()

    outcome = await scrape(
        ScrapeOptions(),
        session_factory=lambda: _session_factory(_StubSession()),
        deps=_deps(_StubRegistry(entries), _StubCrawler(interrupt_after=1), store),
    )

    assert outcome.interrupted is True
    assert outcome.save is None
    assert store.saved == []


async def test_scrape_keeps_completed_batches_when_cancelled():
    entries = [
        _entry("https://www.ird.gov.lk/a"),
        _entry("https://www.ird.gov.lk/b"),
        _entry("https://www.ird.gov.lk/c"),
    ]
    store = _StubStore()
    state = {"stop": False}

    def on_batch(_phase, _batch):
        state["stop"] = True

    outcome = await scrape(
        ScrapeOptions(),
        session_factory=lambda: _session_factory(_StubSession()),
        deps=_deps(_StubRegistry(entries), _StubCrawler(), store),
        on_batch=on_batch,
        should_stop=lambda: state["stop"],
        batch_size=1,
    )

    assert outcome.cancelled is True
    assert outcome.save is not None
    assert outcome.save.saved == 1
    assert len(store.saved) == 1
    assert outcome.crawl.results == []


async def test_get_scrape_status_reads_the_store():
    store = _StubStore()

    status = await get_scrape_status(
        session_factory=lambda: _session_factory(_StubSession()),
        deps=_deps(_StubRegistry([]), _StubCrawler(), store),
    )

    assert status.total == 3


def test_build_scrape_deps_uses_production_adapters():
    deps = build_scrape_deps()

    assert isinstance(deps.registry, URLRegistry)
    assert deps.store_factory is DocumentStore


@pytest.mark.db
async def test_save_results_persists_and_dedups_with_document_store(clean_session):
    url = "https://www.ird.gov.lk/persisted"
    result = AsyncCrawlerFetchResult(
        url=url,
        entry=_entry(url),
        parsed_page=AsyncCrawlerParsedPage(
            title="persisted", text="persisted text", raw_html="", pdf_links=[]
        ),
    )

    @asynccontextmanager
    async def session_factory():
        yield clean_session

    deps = ScrapeDeps(
        registry=_StubRegistry([]),
        crawler_factory=lambda: _StubCrawler(),
        store_factory=DocumentStore,
    )

    first = await save_results([result], full=True, session_factory=session_factory, deps=deps)
    second = await save_results([result], full=False, session_factory=session_factory, deps=deps)

    assert first.saved == 1
    assert second.saved == 0
    assert second.skipped == 1
    count = await clean_session.scalar(select(func.count()).select_from(Document))
    assert count == 1
