"""Authentication: seeded credentials in, a JWT out, one access guard.

There is a single credential path: ``POST /api/v1/auth/token`` exchanges the
seeded username and password hash for a signed access token. Protected routes
accept only ``Authorization: Bearer <token>``.
"""

from __future__ import annotations

import dataclasses
import secrets
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, cast

import jwt
from fastapi import Depends, Request, status
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError
from pwdlib import PasswordHash

from lankatax_ai.backend.api.errors import ApiError
from lankatax_ai.backend.config import Settings

__all__ = [
    "AuthConfig",
    "Principal",
    "get_auth_config",
    "oauth2_scheme",
    "require_access",
]

SUPPORTED_ALGORITHM = "HS256"
MIN_SECRET_BYTES = 32
_ACCESS_TOKEN_TYPE = "access"

_PASSWORD_HASH = PasswordHash.recommended()
# Verified for unknown usernames so both failure paths take the same time.
_DUMMY_HASH = _PASSWORD_HASH.hash(secrets.token_hex(16))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)


@dataclasses.dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated caller behind a request."""

    username: str


@dataclasses.dataclass(frozen=True, slots=True)
class AuthConfig:
    """The seeded credential and the token settings derived from it."""

    username: str
    password_hash: str = dataclasses.field(repr=False)
    secret: str = dataclasses.field(repr=False)
    algorithm: str = SUPPORTED_ALGORITHM
    expire_minutes: int = 30

    @classmethod
    def from_settings(cls, settings: Settings) -> AuthConfig:
        """Build the config, naming every missing value at once."""
        username = settings.api_auth_username
        password_hash = settings.api_auth_password_hash
        secret = settings.api_auth_secret

        missing = [
            label
            for label, value in (
                ("API_AUTH_USERNAME", username),
                ("API_AUTH_PASSWORD_HASH", password_hash),
                ("API_AUTH_SECRET (or API_KEY_TOKEN)", secret),
            )
            if value is None
        ]
        if missing:
            raise RuntimeError(
                "API authentication is enabled but these are unset: " + ", ".join(missing)
            )
        if settings.api_auth_algorithm != SUPPORTED_ALGORITHM:
            raise RuntimeError(
                f"API_AUTH_ALGORITHM must be {SUPPORTED_ALGORITHM}, "
                f"got {settings.api_auth_algorithm}"
            )

        # The checks above guarantee these; the asserts narrow the types.
        assert username is not None
        assert password_hash is not None
        assert secret is not None
        secret_value = secret.get_secret_value()
        if len(secret_value.encode()) < MIN_SECRET_BYTES:
            raise RuntimeError(
                f"API_AUTH_SECRET must be at least {MIN_SECRET_BYTES} bytes; "
                "generate one with `openssl rand -hex 32`"
            )
        return cls(
            username=username,
            password_hash=password_hash.get_secret_value(),
            secret=secret_value,
            algorithm=settings.api_auth_algorithm,
            expire_minutes=settings.api_auth_token_expire_minutes,
        )

    def verify(self, username: str, password: str) -> bool:
        """Compare credentials; unknown users still pay for one hash check."""
        user_ok = secrets.compare_digest(username, self.username)
        expected = self.password_hash if user_ok else _DUMMY_HASH
        password_ok = _PASSWORD_HASH.verify(password, expected)
        return user_ok and password_ok

    def issue(self, username: str) -> tuple[str, int]:
        """Return a signed access token and its lifetime in seconds."""
        now = datetime.now(UTC)
        expires_in = self.expire_minutes * 60
        payload: dict[str, Any] = {
            "sub": username,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=self.expire_minutes)).timestamp()),
            "typ": _ACCESS_TOKEN_TYPE,
        }
        return jwt.encode(payload, self.secret, algorithm=self.algorithm), expires_in

    def decode(self, token: str) -> str | None:
        """Return the subject of a valid access token, otherwise ``None``."""
        try:
            payload = jwt.decode(
                token,
                self.secret,
                algorithms=[self.algorithm],
                options={"require": ["exp", "sub"]},
            )
        except InvalidTokenError:
            return None
        if payload.get("typ") != _ACCESS_TOKEN_TYPE:
            return None
        subject = payload.get("sub")
        if not isinstance(subject, str) or not secrets.compare_digest(subject, self.username):
            return None
        return subject


def get_auth_config(request: Request) -> AuthConfig:
    """Read the config from app state, avoiding a ``deps`` import cycle."""
    return cast(AuthConfig, request.app.state.auth)


async def require_access(
    config: Annotated[AuthConfig, Depends(get_auth_config)],
    token: Annotated[str | None, Depends(oauth2_scheme)],
) -> Principal:
    """Allow the request only with a valid bearer token."""
    if token is not None:
        subject = config.decode(token)
        if subject is not None:
            return Principal(username=subject)
    raise ApiError(
        status_code=status.HTTP_401_UNAUTHORIZED,
        code="unauthorized",
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
