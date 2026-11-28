"""Production wiring for the scrape use case.

The registry is a file-backed adapter, so it is built here (or passed in by the
caller) rather than inside the service. The crawler and store are injected the
same way.
"""

from __future__ import annotations

from lankatax_ai.backend.config import settings
from lankatax_ai.backend.documents import DocumentStore
from lankatax_ai.backend.scraper.crawler import AsyncCrawler
from lankatax_ai.backend.scraper.registry import URLRegistry
from lankatax_ai.backend.services.scrape.types import ScrapeDeps, UrlRegistry


def build_scrape_deps(registry: UrlRegistry | None = None) -> ScrapeDeps:
    """Wire the production registry, crawler and document store."""
    if registry is None:
        registry = URLRegistry()
    return ScrapeDeps(
        registry=registry,
        crawler_factory=AsyncCrawler,
        store_factory=DocumentStore,
        base_host=settings.scrape_base_host,
    )
