"""Scrape use case: options, outcomes, deps and the scrape entry points."""

from __future__ import annotations

from lankatax_ai.backend.services.scrape.deps import build_scrape_deps
from lankatax_ai.backend.services.scrape.service import (
    crawl,
    crawl_batches,
    get_scrape_status,
    list_urls,
    save_results,
    scrape,
    scrape_batches,
)
from lankatax_ai.backend.services.scrape.types import (
    CrawlBatch,
    Crawler,
    CrawlerFactory,
    CrawlOutcome,
    CrawlPhase,
    CrawlSummary,
    SaveOutcome,
    ScrapeBatch,
    ScrapeDeps,
    ScrapeOptions,
    ScrapeOutcome,
    ScrapeStatusRecord,
    ScrapeStore,
    StoreFactory,
    UrlRegistry,
)

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
    "build_scrape_deps",
    "crawl",
    "crawl_batches",
    "get_scrape_status",
    "list_urls",
    "save_results",
    "scrape",
    "scrape_batches",
]
