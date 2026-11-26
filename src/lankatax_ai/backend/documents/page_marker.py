"""The in-band page marker shared by the PDF producer and the chunker.

PDF text is stored as one string with a marker before each page. Both the writer
(the scraper) and the reader (the chunker) import this module, so the format has
a single definition and a single set of helpers.
"""

from __future__ import annotations

import re

PAGE_MARKER = "<!--PAGE:{n}-->"
PAGE_MARKER_PATTERN = re.compile(r"<!--PAGE:(\d+)-->")


def format_page_marker(page: int) -> str:
    """Render the marker that starts a page."""
    return PAGE_MARKER.format(n=page)


def page_numbers(text: str) -> list[int]:
    """Return every page number marked in ``text``, in order."""
    return [int(number) for number in PAGE_MARKER_PATTERN.findall(text)]


def strip_page_markers(text: str) -> str:
    """Remove page markers, leaving the page text."""
    return PAGE_MARKER_PATTERN.sub("", text)


def split_pages(text: str) -> list[tuple[int, str]]:
    """Split marked text into ``(page number, page text)`` pairs.

    Text before the first marker is attached to the first page. Unmarked text is
    treated as a single page numbered 1.
    """
    parts = PAGE_MARKER_PATTERN.split(text)
    pages = [(int(number), body) for number, body in zip(parts[1::2], parts[2::2])]

    leading = parts[0]
    if leading.strip():
        if pages:
            number, body = pages[0]
            pages[0] = (number, leading + body)
        else:
            pages = [(1, leading)]

    return pages
