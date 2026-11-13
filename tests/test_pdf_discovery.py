import pytest

from lankatax_ai.backend.scraper.pdf_discovery import discover_pdf_links
from tests.helpers import make_html_result, make_pdf_result


@pytest.fixture
def registry_urls():
    return {"https://ird.gov.lk/guide.pdf"}


@pytest.fixture
def known_urls():
    return {"https://ird.gov.lk/already-in-db.pdf"}


def test_discovery_pdf_links(registry_urls, known_urls):
    fake_results = [
        make_html_result(
            "https://ird.gov.lk/circulars",
            pdf_links=[
                "https://ird.gov.lk/guide.pdf",  # in registry → skip
                "https://ird.gov.lk/already-in-db.pdf",  # in DB → skip
                "https://ird.gov.lk/brand-new.pdf",  # new → discover
            ],
        )
    ]
    results = discover_pdf_links(fake_results, known_urls, registry_urls)
    assert len(results) == 1


def test_discovery_pdf_empty(registry_urls, known_urls):
    fake_results = [
        make_html_result(
            "https://ird.gov.lk/circulars",
            pdf_links=[
                "https://ird.gov.lk/guide.pdf",  # in registry → skip
                "https://ird.gov.lk/already-in-db.pdf",  # in DB → skip
            ],
        )
    ]
    results = discover_pdf_links(fake_results, known_urls, registry_urls)
    assert len(results) == 0


def test_deduplicated_pdfs(registry_urls, known_urls):
    fake_results = [
        make_html_result(
            "https://ird.gov.lk/circulars",
            pdf_links=[
                "https://ird.gov.lk/guide.pdf",  # in registry → skip
                "https://ird.gov.lk/already-in-db.pdf",  # in DB → skip
                "https://ird.gov.lk/new-pdf.pdf",
                "https://ird.gov.lk/new-pdf.pdf",
            ],
        )
    ]
    results = discover_pdf_links(fake_results, known_urls, registry_urls)
    assert len(results) == 1


def test_pdf_result_skipped(registry_urls, known_urls):
    fake_results = [make_pdf_result("https://ird.gov.lk/some.pdf")]
    result = discover_pdf_links(fake_results, known_urls, registry_urls)
    assert len(result) == 0


def test_empty_registry_urls(known_urls):
    fake_results = [
        make_html_result(
            "https://ird.gov.lk/circulars",
            pdf_links=[
                "https://ird.gov.lk/guide.pdf",  # add
                "https://ird.gov.lk/already-in-db.pdf",  # in DB → skip
                "https://ird.gov.lk/new-pdf.pdf",  # add
            ],
        )
    ]
    result = discover_pdf_links(fake_results, known_urls, {})
    assert len(result) == 2


def test_empty_known_urls(registry_urls):
    fake_results = [
        make_html_result(
            "https://ird.gov.lk/circulars",
            pdf_links=[
                "https://ird.gov.lk/guide.pdf",  # skip
                "https://ird.gov.lk/already-in-db.pdf",  # add
                "https://ird.gov.lk/new-pdf.pdf",  # add
            ],
        )
    ]
    result = discover_pdf_links(fake_results, {}, registry_urls)
    assert len(result) == 2
