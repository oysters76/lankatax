from pathlib import Path

import httpx
import pytest

from lankatax_ai.backend.scraper.crawler import (
    AsyncCrawler,
    AsyncCrawlerErrorKind,
    AsyncCrawlerResult,
)
from tests.helpers import make_scrapable_url_entry

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(
    params=[
        ("income_tax_page.html", "Income tax is charged", "Income Tax"),
        ("vat_page.html", "Chargeability for Value Added Tax excluding", "Value Added Tax (VAT)"),
        (
            "paye_page.html",
            "Employers are required to deduct Income Tax on Employment",
            "Pay As You Earn (PAYE)",
        ),
    ]
)
def parsed_page(request):
    filename, unique_phrase, expected_title = request.param
    html = (FIXTURES / filename).read_text()
    crawler = AsyncCrawler()
    result = crawler.parse_page(html, "https://www.ird.gov.lk")
    return result, unique_phrase, expected_title


def test_noscript_stripped(parsed_page):
    """SharePoint noscript fallback tags should be removed before text extraction."""
    result = parsed_page[0]
    assert "JavaScript enabled" not in result.text


def test_text_not_empty(parsed_page):
    """Parsed page should extract meaningful content, not just empty strings."""
    assert len(parsed_page[0].text) > 100


def test_title_not_empty(parsed_page):
    """Every ird.gov.lk page should have a non-empty title tag."""
    assert len(parsed_page[0].title) > 0


def test_no_duplicate_content(parsed_page):
    """Parent/child selector overlap should not cause text duplication."""
    result, unique_phrase, _ = parsed_page
    assert result.text.count(unique_phrase) == 1


def test_title_extraction(parsed_page):
    """Title should be extracted exactly as it appears in the page's title tag."""
    result, _, expected_title = parsed_page
    assert result.title == expected_title


def test_fallback_to_body():
    """When no content selectors match, parse_page should fall back to body text."""
    html = "<html><body><p>Some tax content here</p></body></html>"
    crawler = AsyncCrawler()
    result = crawler.parse_page(html, "https://www.ird.gov.lk")

    assert "Some tax content here" in result.text


async def test_crawl_all_isolates_parse_failures(monkeypatch):
    crawler = AsyncCrawler()
    good_url = "https://www.ird.gov.lk/good"
    bad_url = "https://www.ird.gov.lk/bad"

    async def fake_fetch(url: str, attempts: int = 3) -> AsyncCrawlerResult:
        return AsyncCrawlerResult(
            url=url,
            status_code=200,
            response=httpx.Response(
                200,
                text="<html><body><p>valid content</p></body></html>",
                request=httpx.Request("GET", url),
            ),
            error=None,
        )

    original_parse = crawler.parse_page

    def parse(html: str, base_url: str):
        if base_url == bad_url:
            raise ValueError("malformed page")
        return original_parse(html, base_url)

    monkeypatch.setattr(crawler, "fetch", fake_fetch)
    monkeypatch.setattr(crawler, "parse_page", parse)

    seen: list[str] = []
    results = await crawler.crawl_all(
        [
            make_scrapable_url_entry(url=good_url, label="good"),
            make_scrapable_url_entry(url=bad_url, label="bad"),
        ],
        on_result=lambda result: seen.append(result.url),
    )

    by_url = {result.url: result for result in results}
    assert len(results) == 2
    assert set(seen) == {good_url, bad_url}
    assert by_url[good_url].parsed_page is not None
    assert by_url[good_url].error is None
    assert by_url[bad_url].parsed_page is None
    assert by_url[bad_url].error is AsyncCrawlerErrorKind.exception
