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


def test_discovery_keeps_only_the_allowed_host():
    fake_results = [
        make_html_result(
            "https://www.ird.gov.lk/circulars",
            pdf_links=[
                "http://dcr-int-ptl-10/en/publications/SEC_2018_07_E.pdf",
                "https://example.com/other.pdf",
                "https://tax.ird.gov.lk/notice.pdf",
                "https://www.ird.gov.lk/included.pdf",
            ],
        )
    ]

    result = discover_pdf_links(fake_results, set(), set(), "ird.gov.lk")

    assert [pdf.url for pdf in result] == [
        "https://tax.ird.gov.lk/notice.pdf",
        "https://www.ird.gov.lk/included.pdf",
    ]


def test_discovery_upgrades_http_and_deduplicates():
    fake_results = [
        make_html_result(
            "https://www.ird.gov.lk/circulars",
            pdf_links=[
                "http://www.ird.gov.lk/same.pdf",
                "https://www.ird.gov.lk/same.pdf",
            ],
        )
    ]

    result = discover_pdf_links(fake_results, set(), set(), "ird.gov.lk")

    assert [pdf.url for pdf in result] == ["https://www.ird.gov.lk/same.pdf"]


def test_discovery_accepts_a_base_url_and_matches_known_https_urls():
    fake_results = [
        make_html_result(
            "https://www.ird.gov.lk/circulars",
            pdf_links=["http://ird.gov.lk/known.pdf", "http://ird.gov.lk/new.pdf"],
        )
    ]

    result = discover_pdf_links(
        fake_results,
        {"https://ird.gov.lk/known.pdf"},
        set(),
        "https://www.ird.gov.lk",
    )

    assert [pdf.url for pdf in result] == ["https://ird.gov.lk/new.pdf"]
