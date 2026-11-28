"""Resolve the caller address without trusting spoofable headers.

``X-Forwarded-For`` is honoured only when the immediate peer is one of the
configured trusted proxies, and then the rightmost untrusted address in the
chain is used. Requests that arrive directly always keep their socket address.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Iterable

__all__ = ["resolve_client_ip"]

_Address = ipaddress.IPv4Address | ipaddress.IPv6Address
_Network = ipaddress.IPv4Network | ipaddress.IPv6Network


def resolve_client_ip(
    peer: str | None,
    forwarded_for: str | None,
    trusted_proxies: Iterable[str],
) -> str:
    """Return the best client address for rate limiting."""
    networks = _parse_networks(trusted_proxies)
    peer_address = _parse_address(peer)

    if (
        peer_address is not None
        and networks
        and _is_trusted(peer_address, networks)
        and forwarded_for
    ):
        chain = [part.strip() for part in forwarded_for.split(",") if part.strip()]
        for candidate in reversed(chain):
            address = _parse_address(candidate)
            if address is not None and not _is_trusted(address, networks):
                return str(address)
        if chain:
            return chain[0]

    if peer_address is not None:
        return str(peer_address)
    return peer or "anonymous"


def _parse_address(value: str | None) -> _Address | None:
    if not value:
        return None
    try:
        return ipaddress.ip_address(value.strip())
    except ValueError:
        return None


def _parse_networks(values: Iterable[str]) -> list[_Network]:
    networks: list[_Network] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned:
            continue
        try:
            networks.append(ipaddress.ip_network(cleaned, strict=False))
        except ValueError:
            continue
    return networks


def _is_trusted(address: _Address, networks: list[_Network]) -> bool:
    return any(address in network for network in networks)
