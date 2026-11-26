"""Compatibility re-exports for the frozen CLI.

Document persistence now lives in :mod:`lankatax_ai.backend.documents`. This
module keeps the old import path working until the CLI is rewritten in Phase 6.
"""

from __future__ import annotations

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
    "ScrapeStatusRecord",
    "get_all_document_ids",
    "get_document_by_id",
    "get_hash_map",
    "get_known_urls",
    "get_pending_document_ids",
    "get_scrape_status",
    "save_documents",
    "update_ingested_hash",
]
