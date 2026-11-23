"""Transport-agnostic use cases shared by the CLI and the API.

Layering rule for this package: it may import ``rag``, ``db``, ``llm`` and
``ingestion``, but never ``typer``, ``rich``, ``fastapi`` or any other transport
library. Rendering and HTTP concerns stay with the caller.
"""

from __future__ import annotations

from lankatax_ai.backend.services.errors import (
    NotFoundError,
    ServiceError,
    UpstreamError,
    ValidationError,
)
from lankatax_ai.backend.services.types import SearchOptions, SearchOutcome

__all__ = [
    "NotFoundError",
    "SearchOptions",
    "SearchOutcome",
    "ServiceError",
    "UpstreamError",
    "ValidationError",
]
