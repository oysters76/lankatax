from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import Select

from lankatax_ai.backend.db.models import Chunk, Document


@dataclasses.dataclass
class MetadataFilters:
    """Structured filters extracted from CLI flags or agent intent detection."""

    category: str | None = None
    doc_type: str | None = None
    after: date | None = None
    before: date | None = None

    @property
    def has_any(self) -> bool:
        return any([self.category, self.doc_type, self.after, self.before])


def apply_metadata_filters[RowT: tuple[Any, ...]](
    stmt: Select[RowT], filters: MetadataFilters
) -> Select[RowT]:
    """Apply metadata WHERE clauses to an existing SQLAlchemy select statement.

    Joins chunks → documents when any filter is set, then adds WHERE clauses.
    """
    if not filters.has_any:
        return stmt

    stmt = stmt.join(Document, Chunk.document_id == Document.id)

    if filters.category:
        stmt = stmt.where(Document.category == filters.category)

    if filters.doc_type:
        stmt = stmt.where(Document.document_kind == filters.doc_type)

    if filters.after:
        after_dt = datetime(filters.after.year, filters.after.month, filters.after.day, tzinfo=UTC)
        stmt = stmt.where(Document.scraped_at >= after_dt)

    if filters.before:
        before_dt = datetime(
            filters.before.year, filters.before.month, filters.before.day, 23, 59, 59, tzinfo=UTC
        )
        stmt = stmt.where(Document.scraped_at <= before_dt)

    return stmt
