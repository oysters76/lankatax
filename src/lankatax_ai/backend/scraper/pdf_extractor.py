import io
import re

import pdfplumber

PAGE_MARKER = "<!--PAGE:{n}-->"
PAGE_MARKER_PATTERN = re.compile(r"<!--PAGE:(\d+)-->")


def safe_str(s: str | None) -> str:
    return "" if s is None else s


def pdf_to_text(pdf_bytes: bytes) -> str:
    text: str = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_text = PAGE_MARKER.format(n=i)
            page_text += safe_str(page.extract_text())
            text += page_text
    return text
