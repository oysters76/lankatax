from pathlib import Path
import pytest

from lankatax_ai.backend.scraper.crawler import AsyncCrawler

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
