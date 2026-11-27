"""Result types produced by the scraper adapters."""

from __future__ import annotations

import dataclasses
from enum import Enum

import httpx

from lankatax_ai.backend.scraper.registry import ScrapableURL

__all__ = [
    "AsyncCrawlerErrorKind",
    "AsyncCrawlerFetchResult",
    "AsyncCrawlerParsedPage",
    "AsyncCrawlerResult",
]


class AsyncCrawlerErrorKind(Enum):
    not_found = "not_found"
    server_error = "server_error"
    no_permission = "no_permission"
    exception = "exception"
    retries_exhausted = "retries_exhausted"
    unknown_error = "unknown_error"


@dataclasses.dataclass
class AsyncCrawlerParsedPage:
    title: str
    text: str
    raw_html: str
    pdf_links: list[str]


@dataclasses.dataclass
class AsyncCrawlerResult:
    url: str
    status_code: int
    response: httpx.Response | None
    error: AsyncCrawlerErrorKind | None

    @property
    def is_error(self) -> bool:
        return self.error is not None


@dataclasses.dataclass
class AsyncCrawlerFetchResult:
    url: str
    entry: ScrapableURL
    parsed_page: AsyncCrawlerParsedPage | None = None
    pdf_bytes: bytes | None = None
    error: AsyncCrawlerErrorKind | None = None

    @property
    def is_error(self) -> bool:
        return self.error is not None
