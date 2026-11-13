from lankatax_ai.backend.scraper.pdf_extractor import PAGE_MARKER, PAGE_MARKER_PATTERN


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
