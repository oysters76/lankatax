from lankatax_ai.backend.scraper.crawler import (
    AsyncCrawlerFetchResult,
    AsyncCrawlerParsedPage,
)
from lankatax_ai.backend.scraper.registry import ScrapableURL, URLCategory
from lankatax_ai.backend.scraper.storage import DocumentRecord
from lankatax_ai.backend.types import DocumentKind


def make_scrapable_url_entry(**overrides) -> ScrapableURL:
    defaults = {
        "url": "https://www.ird.gov.lk/test",
        "category": URLCategory.tax_info,
        "label": "Test Page",
        "kind": DocumentKind.page,
    }
    return ScrapableURL(**(defaults | overrides))


def make_document_record(**overrides) -> DocumentRecord:
    defaults = {
        "url": "https://www.ird.gov.lk/test",
        "title": "Test Page",
        "raw_text": "Some tax content",
        "category": "tax_info",
        "content_hash": "abc123",
        "raw_html": "",
        "document_kind": "page",
    }
    return DocumentRecord(**(defaults | overrides))


def make_html_result(url, pdf_links=None, category=URLCategory.tax_info):
    return AsyncCrawlerFetchResult(
        url=url,
        entry=ScrapableURL(url=url, category=category, label="Test", kind=DocumentKind.page),
        parsed_page=AsyncCrawlerParsedPage(
            title="Test", text="content", raw_html="<p>content</p>", pdf_links=pdf_links or []
        ),
    )


def make_pdf_result(url, pdf_bytes=b"%PDF-fake", category=URLCategory.tax_info):
    return AsyncCrawlerFetchResult(
        url=url,
        entry=ScrapableURL(url=url, category=category, label="Test PDF", kind=DocumentKind.pdf),
        pdf_bytes=pdf_bytes,
    )
