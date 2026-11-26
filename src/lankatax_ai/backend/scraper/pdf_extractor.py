import io

import pdfplumber

from lankatax_ai.backend.documents.page_marker import (
    PAGE_MARKER,
    PAGE_MARKER_PATTERN,
    format_page_marker,
)

__all__ = ["PAGE_MARKER", "PAGE_MARKER_PATTERN", "format_page_marker", "pdf_to_text", "safe_str"]


def safe_str(s: str | None) -> str:
    return "" if s is None else s


def pdf_to_text(pdf_bytes: bytes) -> str:
    text: str = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_text = format_page_marker(i)
            page_text += safe_str(page.extract_text())
            text += page_text
    return text
