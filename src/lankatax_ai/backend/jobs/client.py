"""Hatchet client construction.

The client is built lazily by callers; importing this module never connects to
Hatchet or requires a token.
"""

from __future__ import annotations

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

    tls_strategy = active.hatchet_client_tls_strategy
    host_port = active.hatchet_client_host_port
    token_value = token.get_secret_value()
    tls_config = ClientTLSConfig(strategy=tls_strategy) if tls_strategy is not None else None

    if host_port is None:
        if tls_config is None:
            client_config = ClientConfig(token=token_value)
        else:
            client_config = ClientConfig(token=token_value, tls_config=tls_config)
    elif tls_config is None:
        client_config = ClientConfig(token=token_value, host_port=host_port)
    else:
        client_config = ClientConfig(
            token=token_value,
            host_port=host_port,
            tls_config=tls_config,
        )

    return Hatchet(config=client_config)
