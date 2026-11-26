"""Shared document contracts and persistence.

The ``documents`` table is the boundary between scraping and ingestion: scraping
writes documents, ingestion reads them and records ingestion state. This package
owns that boundary so the two pipelines stay peers instead of importing each
other's internals.
"""

from __future__ import annotations

from lankatax_ai.backend.documents.page_marker import (
    PAGE_MARKER,
    PAGE_MARKER_PATTERN,
    format_page_marker,
    page_numbers,
    split_pages,
    strip_page_markers,
)
from lankatax_ai.backend.documents.store import (
    ScrapeStatusRecord,
    get_all_document_ids,
    get_document_by_id,
    get_hash_map,
    get_known_urls,
    get_pending_document_ids,
    get_scrape_status,
    save_documents,
    update_ingested_hash,
)

__all__ = [
    "PAGE_MARKER",
    "PAGE_MARKER_PATTERN",
    "ScrapeStatusRecord",
    "format_page_marker",
    "get_all_document_ids",
    "get_document_by_id",
    "get_hash_map",
    "get_known_urls",
    "get_pending_document_ids",
    "get_scrape_status",
    "page_numbers",
    "save_documents",
    "split_pages",
    "strip_page_markers",
    "update_ingested_hash",
]
