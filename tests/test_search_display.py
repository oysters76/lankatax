from __future__ import annotations

from uuid import uuid4

import pytest
from rich.console import Console

from lankatax_ai.backend.rag.types import RetrievalResult
from lankatax_ai.cli.formatting import format_sources, print_sources

pytestmark = pytest.mark.unit

URL = "https://www.ird.gov.lk/en/APIT_Tax_Tables_2025.pdf"


def _result(
    *,
    title: str | None = "APIT Tax Tables",
    url: str | None = URL,
) -> RetrievalResult:
    metadata: dict[str, str] = {}
    if title is not None:
        metadata["document_title"] = title
    if url is not None:
        metadata["from_url"] = url
    return RetrievalResult(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content="APIT is calculated on monthly remuneration...",
        score=0.87,
        vector_score=0.8,
        keyword_score=0.6,
        metadata=metadata,
    )


def _console(width: int = 80) -> Console:
    return Console(width=width, record=True, force_terminal=False, color_system=None)


def test_sources_are_numbered_and_paired_with_urls():
    results = [_result(), _result(title="PAYE Guide", url="https://ird.gov.lk/paye.pdf")]

    lines = format_sources(results)

    assert lines == [
        f"  [1] APIT Tax Tables — {URL}",
        "  [2] PAYE Guide — https://ird.gov.lk/paye.pdf",
    ]


def test_missing_metadata_falls_back_to_placeholders():
    lines = format_sources([_result(title=None, url=None)])

    assert lines == ["  [1] Untitled — no URL"]


def test_print_sources_keeps_long_url_on_one_line():
    console = _console(width=40)

    print_sources([_result()], console)

    text = console.export_text()
    assert f"[1] APIT Tax Tables — {URL}" in text
    assert URL in text  # not folded mid-URL despite the 40-char console


def test_print_sources_is_silent_without_results():
    console = _console()

    print_sources([], console)

    assert console.export_text() == ""
