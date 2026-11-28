"""One error shape for the whole API, and the handlers that produce it.

Services raise :class:`~lankatax_ai.backend.services.errors.ServiceError`; the
API is the only layer that turns those (or its own auth failures) into JSON.
Every error carries ``links`` so a client can recover without guessing URLs.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from lankatax_ai.backend.api.utils import error_payload
from lankatax_ai.backend.services.errors import RateLimitError, ServiceError

__all__ = ["ApiError", "install_error_handlers"]

STATUS_BY_SERVICE_CODE: dict[str, int] = {
    "validation_error": 422,
    "not_found": 404,
    "rate_limited": 429,
    "upstream_error": 502,
}

HTTP_CODE_BY_STATUS: dict[int, str] = {
    404: "not_found",
    405: "method_not_allowed",
}


class ApiError(Exception):
    """An HTTP-layer failure (auth, health, stubs); never raised by services."""

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        detail: str,
        errors: list[dict[str, Any]] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.code = code
        self.detail = detail
        self.errors = errors
        self.headers = headers


def install_error_handlers(app: FastAPI) -> None:
    """Register every handler; all of them emit the hypermedia envelope."""

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _response(
            request,
            status_code=exc.status_code,
            code=exc.code,
            detail=exc.detail,
            errors=exc.errors,
            headers=exc.headers,
        )

    @app.exception_handler(ServiceError)
    async def _service_error(request: Request, exc: ServiceError) -> JSONResponse:
        headers = None
        if isinstance(exc, RateLimitError) and exc.retry_after is not None:
            headers = {"Retry-After": str(exc.retry_after)}
        return _response(
            request,
            status_code=STATUS_BY_SERVICE_CODE.get(exc.code, 500),
            code=exc.code,
            detail=str(exc),
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _response(
            request,
            status_code=422,
            code="validation_error",
            detail="Request validation failed",
            errors=[dict(error) for error in exc.errors()],
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _response(
            request,
            status_code=exc.status_code,
            code=HTTP_CODE_BY_STATUS.get(exc.status_code, "http_error"),
            detail=str(exc.detail),
            headers=dict(exc.headers) if exc.headers else None,
        )


def _response(
    request: Request,
    *,
    status_code: int,
    code: str,
    detail: str,
    errors: list[dict[str, Any]] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = error_payload(request, code=code, detail=detail, errors=errors)
    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder(body),
        headers=headers,
    )
