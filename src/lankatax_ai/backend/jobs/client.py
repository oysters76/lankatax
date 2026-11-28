"""Hatchet client construction.

The client is built lazily by callers; importing this module never connects to
Hatchet or requires a token.
"""

from __future__ import annotations

from typing import Any

from hatchet_sdk import Hatchet
from hatchet_sdk.config import ClientConfig, ClientTLSConfig

from lankatax_ai.backend.config import Settings, settings

__all__ = ["build_hatchet_client"]


def build_hatchet_client(config: Settings | None = None) -> Hatchet:
    """Build a Hatchet client from settings.

    Raises ``RuntimeError`` when the token is missing or blank; the SDK defaults
    are intentionally not allowed to create a silently broken client.
    """
    active = config or settings
    token = active.hatchet_client_token
    if token is None or not token.get_secret_value().strip():
        raise RuntimeError("HATCHET_CLIENT_TOKEN is required to build the Hatchet client")

    # Only pass values that were configured: the SDK otherwise reads the gRPC
    # and REST addresses from the token, which is right for Hatchet Cloud.
    overrides: dict[str, Any] = {}
    if active.hatchet_client_server_url is not None:
        overrides["server_url"] = active.hatchet_client_server_url
    if active.hatchet_client_host_port is not None:
        overrides["host_port"] = active.hatchet_client_host_port
    if active.hatchet_client_tls_strategy is not None:
        overrides["tls_config"] = ClientTLSConfig(strategy=active.hatchet_client_tls_strategy)

    # Our Settings already read .env; do not let the SDK read it a second time.
    client_config = ClientConfig(
        _env_file=None,  # type: ignore[call-arg]  # pydantic-settings runtime option
        token=token.get_secret_value(),
        **overrides,
    )
    return Hatchet(config=client_config)
