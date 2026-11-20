from __future__ import annotations

from rich.console import Console

from lankatax_ai.backend.rag.types import RetrievalResult


def format_sources(results: list[RetrievalResult]) -> list[str]:
    """One numbered ``[n] title — url`` line per result, aligned with table ranks."""
    lines = []
    for i, r in enumerate(results, 1):
        title = r.metadata.get("document_title") or "Untitled"
        url = r.metadata.get("from_url") or "no URL"
        lines.append(f"  [{i}] {title} — {url}")
    return lines


def print_sources(results: list[RetrievalResult], console: Console) -> None:
    if not results:
        return
    console.print("\n[bold]Sources[/bold]")
    for line in format_sources(results):
        # soft_wrap keeps long URLs on one line so they stay copy-pasteable.
        console.print(line, markup=False, highlight=False, soft_wrap=True)
