"""Token issuing: seeded username and password in, a bearer token out."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request, status
from fastapi.security import OAuth2PasswordRequestForm

from lankatax_ai.backend.api.deps import ApiDepsDep
from lankatax_ai.backend.api.errors import ApiError
from lankatax_ai.backend.api.schemas import TokenResponse
from lankatax_ai.backend.api.utils import link

router = APIRouter(tags=["auth"])


@router.post(
    "/api/v1/auth/token",
    response_model=TokenResponse,
    summary="Exchange the seeded credentials for an access token",
)
async def token(
    request: Request,
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    deps: ApiDepsDep,
) -> TokenResponse:
    if not deps.auth.verify(form.username, form.password):
        raise ApiError(
            status_code=status.HTTP_401_UNAUTHORIZED,
            code="unauthorized",
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token, expires_in = deps.auth.issue(form.username)
    return TokenResponse(
        access_token=access_token,
        expires_in=expires_in,
        links={
            "self": link(request, "token", method="POST"),
            "api_root": link(request, "api_root"),
            "search": link(request, "search", method="POST"),
            "stats": link(request, "stats"),
            "cache_stats": link(request, "cache_stats"),
        },
    )
