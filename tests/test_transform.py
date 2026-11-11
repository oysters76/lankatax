from unittest.mock import patch

from lankatax_ai.backend.scraper.content_hasher import compute_hash
from lankatax_ai.backend.scraper.crawler import (
    AsyncCrawlerErrorKind,
    AsyncCrawlerFetchResult,
    AsyncCrawlerParsedPage,
)
from lankatax_ai.backend.scraper.registry import DocumentKind, ScrapableURL, URLCategory
from lankatax_ai.backend.scraper.transform import to_document_record


def _make_entry(**overrides) -> ScrapableURL:
    defaults = {
        "url": "https://www.ird.gov.lk/test",
        "category": URLCategory.tax_info,
        "label": "Test Page",
        "kind": DocumentKind.page,
    }
    return ScrapableURL(**(defaults | overrides))


def test_error_result_returns_none():
    """Crawl errors (404, timeout, etc.) should produce no document record."""
    result = AsyncCrawlerFetchResult(
        url="https://www.ird.gov.lk/missing",
        entry=_make_entry(),
        error=AsyncCrawlerErrorKind.not_found,
    )
    assert to_document_record(result) is None


def test_no_page_no_pdf_returns_none():
    """A result with neither parsed page nor PDF bytes should return None."""
    result = AsyncCrawlerFetchResult(
        url="https://www.ird.gov.lk/empty",
        entry=_make_entry(),
    )
    assert to_document_record(result) is None


def test_page_result_converts():
    """HTML page result should map to a DocumentRecord with correct fields and hash."""
    page = AsyncCrawlerParsedPage(
        title="Income Tax",
        text="Income tax is charged on all persons.",
        raw_html="<div>Income tax is charged on all persons.</div>",
        pdf_links=[],
    )
    result = AsyncCrawlerFetchResult(
        url="https://www.ird.gov.lk/income-tax",
        entry=_make_entry(),
        parsed_page=page,
    )
    record = to_document_record(result)

    assert record is not None
    assert record.url == "https://www.ird.gov.lk/income-tax"
    assert record.title == "Income Tax"
    assert record.raw_text == "Income tax is charged on all persons."
    assert record.category == "tax_info"
    assert record.content_hash == compute_hash("Income tax is charged on all persons.")


@patch("lankatax_ai.backend.scraper.transform.pdf_to_text", return_value="Extracted PDF text")
def test_pdf_result_converts(mock_pdf):
    """PDF result should use entry label as title and hash the raw bytes."""
    pdf_bytes = b"%PDF-fake-content"
    result = AsyncCrawlerFetchResult(
        url="https://www.ird.gov.lk/guide.pdf",
        entry=_make_entry(kind=DocumentKind.pdf, label="Tax Guide"),
        pdf_bytes=pdf_bytes,
    )
    record = to_document_record(result)

    assert record is not None
    assert record.url == "https://www.ird.gov.lk/guide.pdf"
    assert record.title == "Tax Guide"
    assert record.raw_text == "Extracted PDF text"
    assert record.content_hash == compute_hash(pdf_bytes)
    mock_pdf.assert_called_once_with(pdf_bytes)


def test_page_takes_priority_over_pdf():
    """When both parsed_page and pdf_bytes exist, the page should be used."""
    page = AsyncCrawlerParsedPage(
        title="Page Title",
        text="Page content",
        raw_html="<p>Page content</p>",
        pdf_links=[],
    )
    result = AsyncCrawlerFetchResult(
        url="https://www.ird.gov.lk/both",
        entry=_make_entry(),
        parsed_page=page,
        pdf_bytes=b"%PDF-also-here",
    )
    record = to_document_record(result)

    assert record is not None
    assert record.title == "Page Title"
    assert record.raw_text == "Page content"
