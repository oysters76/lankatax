from __future__ import annotations

import asyncio
import dataclasses
import random
import re
from collections.abc import Callable
from enum import Enum
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup as bs
from httpx import Timeout

from lankatax_ai.backend.scraper.registry import DocumentKind, ScrapableURL


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
    def is_error(self):
        return self.error is not None


@dataclasses.dataclass
class AsyncCrawlerFetchResult:
    url: str
    entry: ScrapableURL
    parsed_page: AsyncCrawlerParsedPage | None = None
    pdf_bytes: bytes | None = None
    error: AsyncCrawlerErrorKind | None = None

    @property
    def is_error(self):
        return self.error is not None


class AsyncCrawler:
    CONTENT_SELECTORS = [
        "#ctl00_PlaceHolderMain",
        ".ms-rtestate-field",
        "#contentBox",
        ".ms-WPBody",
    ]

    def __init__(
        self,
        timeout: float = 10.0,
        connect: int = 5,
        retries: int = 3,
        headers: dict[str:str] = {"User-Agent": "LankaTaxAI"},
        min_delay: float = 3,
        max_delay: float = 5,
    ):
        self._timeout_size = timeout
        self._connect = connect
        self._retries = retries
        self._headers = headers
        self._timeout: httpx.Timeout | None = None
        self._transport: httpx.AsyncHTTPTransport | None = None
        self._client: httpx.AsyncClient | None = None
        self._lock: asyncio.Lock = asyncio.Lock()
        self._min_delay = min_delay
        self._max_delay = max_delay

        self._regex_heading = re.compile(r"^h[1-6]$")

    async def __aenter__(self) -> AsyncCrawler:
        self._timeout = Timeout(self._timeout_size, connect=self._connect)
        self._transport = httpx.AsyncHTTPTransport(retries=self._retries)
        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            transport=self._transport,
            headers=self._headers,
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._client is None:
            return
        if self._client.is_closed:
            return
        await self._client.aclose()
        self._client = None

    async def _rate_limit(self):
        async with self._lock:
            await asyncio.sleep(random.uniform(self._min_delay, self._max_delay))

    async def _wait_longer(self, attempt: int):
        await asyncio.sleep(2 ** (attempt + 2))

    async def _wait_shorter(self, attempt: int):
        await asyncio.sleep(2**attempt)

    async def fetch(self, url: str, attempts: int = 3) -> AsyncCrawlerResult:
        try:
            for attempt in range(attempts):
                await self._rate_limit()
                response = await self._client.get(url)
                sc = response.status_code

                if sc in (200, 201):
                    return AsyncCrawlerResult(
                        url=url, status_code=sc, response=response, error=None
                    )
                elif sc == 404:
                    return AsyncCrawlerResult(
                        url=url,
                        status_code=sc,
                        response=None,
                        error=AsyncCrawlerErrorKind.not_found,
                    )
                elif sc == 401:
                    return AsyncCrawlerResult(
                        url=url,
                        status_code=sc,
                        response=None,
                        error=AsyncCrawlerErrorKind.no_permission,
                    )
                elif sc == 429:
                    await self._wait_longer(attempt)
                elif 500 <= sc <= 503:
                    await self._wait_shorter(attempt)
                else:
                    return AsyncCrawlerResult(
                        url=url,
                        status_code=-1,
                        response=None,
                        error=AsyncCrawlerErrorKind.unknown_error,
                    )

            return AsyncCrawlerResult(
                url=url,
                status_code=-1,
                response=None,
                error=AsyncCrawlerErrorKind.retries_exhausted,
            )
        except Exception:
            return AsyncCrawlerResult(
                url=url, status_code=-1, response=None, error=AsyncCrawlerErrorKind.exception
            )

    def parse_page(
        self, html: str, base_url: str, selectors: list[str] = CONTENT_SELECTORS
    ) -> AsyncCrawlerParsedPage:
        soup = bs(html, "html.parser")

        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # Collect content blocks from selectors
        parts_text: list[str] = []
        parts_html: list[str] = []
        seen: set[int] = set()

        for selector in selectors:
            for el in soup.select(selector):
                if id(el) in seen:
                    continue
                seen.add(id(el))
                text = el.get_text(separator="\n", strip=True)
                if text:
                    parts_text.append(text)
                    parts_html.append(str(el))

        # Fallback to body
        if not parts_text:
            body = soup.find("body") or soup
            body_text = body.get_text(separator="\n", strip=True)
            if body_text:
                parts_text.append(body_text)
                parts_html.append(str(body))

        # Collect PDF links
        pdf_links = [
            urljoin(base_url, a["href"])
            for a in soup.find_all("a", href=True)
            if a["href"].lower().endswith(".pdf")
        ]

        return AsyncCrawlerParsedPage(
            title=title,
            text="\n\n".join(parts_text),
            raw_html="\n".join(parts_html),
            pdf_links=pdf_links,
        )

    async def crawl_url(self, entry: ScrapableURL) -> AsyncCrawlerFetchResult:
        result = await self.fetch(entry.url)

        if result.is_error:
            return AsyncCrawlerFetchResult(url=entry.url, entry=entry, error=result.error)

        if entry.kind == DocumentKind.pdf:
            return AsyncCrawlerFetchResult(
                url=entry.url,
                entry=entry,
                pdf_bytes=result.response.content,
            )

        parsed = self.parse_page(result.response.text, entry.url)
        return AsyncCrawlerFetchResult(
            url=entry.url,
            entry=entry,
            parsed_page=parsed,
        )

    async def crawl_all(
        self,
        entries: list[ScrapableURL],
        max_concurrent: int = 5,
        on_result: Callable[[AsyncCrawlerFetchResult], None] | None = None,
    ) -> list[AsyncCrawlerFetchResult]:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _limited(entry: ScrapableURL) -> AsyncCrawlerFetchResult:
            async with semaphore:
                result = await self.crawl_url(entry)
                if on_result:
                    on_result(result)
                return result

        return await asyncio.gather(*[_limited(e) for e in entries])
