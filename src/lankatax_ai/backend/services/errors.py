"""Transport-agnostic service failures.

Services must raise these instead of framework exceptions (``typer.BadParameter``,
``fastapi.HTTPException``). Only the transport renders them: the API maps each
``code`` to an HTTP status and a JSON body; the CLI reads that JSON and never
sees these classes.
"""

from __future__ import annotations

from typing import ClassVar


class ServiceError(Exception):
    """Base class for use-case failures."""

    code: ClassVar[str] = "service_error"


class ValidationError(ServiceError):
    """The caller supplied invalid input or options."""

    code: ClassVar[str] = "validation_error"


class NotFoundError(ServiceError):
    """The requested resource does not exist."""

    code: ClassVar[str] = "not_found"


class UpstreamError(ServiceError):
    """A dependency (LLM, embeddings, database) failed."""

    code: ClassVar[str] = "upstream_error"


class RateLimitError(ServiceError):
    """The caller exceeded an allowed request rate."""

    code: ClassVar[str] = "rate_limited"

    def __init__(
        self, message: str = "Too many requests", *, retry_after: int | None = None
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
