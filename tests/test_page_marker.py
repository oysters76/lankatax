from lankatax_ai.backend.documents.page_marker import (
    PAGE_MARKER,
    PAGE_MARKER_PATTERN,
    format_page_marker,
    page_numbers,
    split_pages,
    strip_page_markers,
)
from lankatax_ai.backend.ingestion.chunker.structural import chunk_structural
from lankatax_ai.backend.ingestion.types import ChunkConfig
from tests.helpers import make_document


def test_pattern_matches_marker_single_digit():
    """Single-digit page number formatted by PAGE_MARKER must match PAGE_MARKER_PATTERN."""
    marker = PAGE_MARKER.format(n=1)
    assert PAGE_MARKER_PATTERN.search(marker) is not None


def test_pattern_matches_marker_multi_digit():
    """Multi-digit page numbers (10, 99, 100) must all match PAGE_MARKER_PATTERN."""
    for n in [10, 99, 100]:
        marker = PAGE_MARKER.format(n=n)
        assert PAGE_MARKER_PATTERN.search(marker) is not None


def test_pattern_extracts_page_number():
    """PAGE_MARKER_PATTERN.group(1) must return the page number as a string."""
    marker = PAGE_MARKER.format(n=42)
    match = PAGE_MARKER_PATTERN.search(marker)
    assert match is not None
    assert match.group(1) == "42"


def test_pattern_finds_all_markers_in_text():
    """findall must extract every page number from text with multiple markers."""
    text = PAGE_MARKER.format(n=1) + "First page." + PAGE_MARKER.format(n=2) + "Second page."
    pages = PAGE_MARKER_PATTERN.findall(text)
    assert pages == ["1", "2"]


def test_marker_and_pattern_stay_in_sync():
    """Contract test: PAGE_MARKER and PAGE_MARKER_PATTERN must agree for all page numbers."""
    for n in range(1, 201):
        marker = PAGE_MARKER.format(n=n)
        match = PAGE_MARKER_PATTERN.search(marker)
        assert match is not None, f"PAGE_MARKER_PATTERN failed to match PAGE_MARKER for page {n}"
        assert match.group(1) == str(n), f"Extracted {match.group(1)} but expected {n}"


def test_split_pages_returns_numbers_and_bodies():
    text = format_page_marker(1) + "First." + format_page_marker(2) + "Second."

    assert split_pages(text) == [(1, "First."), (2, "Second.")]


def test_split_pages_treats_unmarked_text_as_page_one():
    assert split_pages("No markers here.") == [(1, "No markers here.")]


def test_split_pages_attaches_leading_text_to_first_page():
    text = "Preamble." + format_page_marker(1) + "First."

    assert split_pages(text) == [(1, "Preamble.First.")]


def test_page_numbers_and_strip_agree():
    text = format_page_marker(1) + "First." + format_page_marker(7) + "Second."

    assert page_numbers(text) == [1, 7]
    assert strip_page_markers(text) == "First.Second."


def test_structural_pdf_chunks_keep_page_citations_and_clean_content():
    doc = make_document(
        document_kind="pdf",
        raw_text=format_page_marker(1)
        + "First page text."
        + format_page_marker(2)
        + "Second page text.",
        raw_html="",
    )

    chunks = chunk_structural(doc, ChunkConfig())

    assert chunks
    assert all("<!--PAGE" not in chunk.content for chunk in chunks)
    assert all("-->" not in chunk.content for chunk in chunks)
    assert chunks[0].citation.page_start == 1
    assert chunks[0].citation.page_end == 2
