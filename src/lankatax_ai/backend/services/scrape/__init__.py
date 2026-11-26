"""Scrape use case: options, outcomes, deps and the scrape entry points."""

from __future__ import annotations

from lankatax_ai.backend.services.scrape.deps import build_scrape_deps
from lankatax_ai.backend.services.scrape.service import (
    crawl,
    get_scrape_status,
    list_urls,
    save_results,
    scrape,
)
from lankatax_ai.backend.services.scrape.types import (
    Crawler,
    CrawlerFactory,
    CrawlOutcome,
    CrawlPhase,
    ProgressCallback,
    SaveOutcome,
    ScrapeDeps,
    ScrapeOptions,
    ScrapeOutcome,
    ScrapeStatusRecord,
    ScrapeStore,
    StoreFactory,
    UrlRegistry,
)

__all__ = [
    "CrawlOutcome",
    "CrawlPhase",
    "Crawler",
    "CrawlerFactory",
    "ProgressCallback",
    "SaveOutcome",
    "ScrapeDeps",
    "ScrapeOptions",
    "ScrapeOutcome",
    "ScrapeStatusRecord",
    "ScrapeStore",
    "StoreFactory",
    "UrlRegistry",
    "build_scrape_deps",
    "crawl",
    "get_scrape_status",
    "list_urls",
    "save_results",
    "scrape",
]
