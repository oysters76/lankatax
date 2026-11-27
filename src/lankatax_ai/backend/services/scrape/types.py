"""Types for the scrape use case.

The crawl result and status types live with the scraper and the documents store,
because those adapters and their callers share them. This module adds the
request shape, outcomes, ports and wiring the service needs.
"""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from enum import Enum
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from lankatax_ai.backend.db.models import Document
from lankatax_ai.backend.documents import ScrapeStatusRecord
from lankatax_ai.backend.scraper.crawler import AsyncCrawlerFetchResult
from lankatax_ai.backend.scraper.registry import ScrapableURL

__all__ = [
    "CrawlBatch",
    "CrawlOutcome",
    "CrawlPhase",
    "CrawlSummary",
    "Crawler",
    "CrawlerFactory",
    "SaveOutcome",
    "ScrapeBatch",
    "ScrapeDeps",
    "ScrapeOptions",
    "ScrapeOutcome",
    "ScrapeStatusRecord",
    "ScrapeStore",
    "StoreFactory",
    "UrlRegistry",
]


class CrawlPhase(Enum):
    registry = "registry"
    discovered = "discovered"


class Crawler(Protocol):
    def crawl_batches(
        self,
        entries: list[ScrapableURL],
        max_concurrent: int = 5,
        batch_size: int = 25,
    ) -> AsyncIterator[list[AsyncCrawlerFetchResult]]: ...


class UrlRegistry(Protocol):
    def get_all(self) -> list[ScrapableURL]: ...


class ScrapeStore(Protocol):
    async def known_urls(self) -> set[str]: ...

    async def hash_map(self, urls: list[str] | None = None) -> dict[str, str]: ...

    async def save(self, documents: list[Document]) -> None: ...

    async def scrape_status(self) -> ScrapeStatusRecord: ...

    async def commit(self) -> None: ...


CrawlerFactory = Callable[[], AbstractAsyncContextManager[Crawler]]
StoreFactory = Callable[[AsyncSession], ScrapeStore]


@dataclasses.dataclass(frozen=True, slots=True)
class ScrapeOptions:
    """What to crawl and whether to force a re-save."""

    full: bool = False
    discover_pdfs: bool = True


@dataclasses.dataclass(frozen=True, slots=True)
class CrawlBatch:
    """One completed unit of crawl work."""

    phase: CrawlPhase
    results: list[AsyncCrawlerFetchResult]
    discovered: tuple[ScrapableURL, ...] = ()


@dataclasses.dataclass(slots=True)
class CrawlSummary:
    succeeded: int
    failed: int
    discovered: int
    errors: list[str] = dataclasses.field(default_factory=list)
    interrupted: bool = False


@dataclasses.dataclass(slots=True)
class CrawlOutcome(CrawlSummary):
    results: list[AsyncCrawlerFetchResult] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True, slots=True)
class SaveOutcome:
    saved: int
    skipped: int


@dataclasses.dataclass(frozen=True, slots=True)
class ScrapeBatch:
    """A crawl batch that has already been persisted."""

    crawl: CrawlBatch
    save: SaveOutcome


@dataclasses.dataclass(slots=True)
class ScrapeOutcome:
    crawl: CrawlSummary
    save: SaveOutcome | None = None

    @property
    def interrupted(self) -> bool:
        return self.crawl.interrupted


@dataclasses.dataclass(frozen=True, slots=True)
class ScrapeDeps:
    """Collaborators the scrape use case needs, wired by the caller."""

    registry: UrlRegistry
    crawler_factory: CrawlerFactory
    store_factory: StoreFactory
