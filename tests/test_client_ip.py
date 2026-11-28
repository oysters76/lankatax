from __future__ import annotations

import pytest

from lankatax_ai.backend.api.client_ip import resolve_client_ip

pytestmark = pytest.mark.unit


def test_direct_peer_wins_and_forwarded_header_is_ignored():
    assert resolve_client_ip("203.0.113.9", "1.2.3.4", []) == "203.0.113.9"
    assert resolve_client_ip("203.0.113.9", "1.2.3.4", ["10.0.0.0/8"]) == "203.0.113.9"


def test_trusted_proxy_uses_the_rightmost_untrusted_address():
    assert (
        resolve_client_ip("10.0.0.2", "203.0.113.9, 198.51.100.7", ["10.0.0.0/8"]) == "198.51.100.7"
    )


def test_multiple_trusted_proxies_are_skipped():
    assert resolve_client_ip("10.0.0.2", "198.51.100.7, 10.1.2.3", ["10.0.0.0/8"]) == "198.51.100.7"


def test_all_forwarded_entries_trusted_falls_back_to_the_first():
    assert resolve_client_ip("10.0.0.2", "10.1.1.1, 10.2.2.2", ["10.0.0.0/8"]) == "10.1.1.1"


def test_invalid_forwarded_entries_are_ignored():
    assert (
        resolve_client_ip("10.0.0.2", "not-an-ip, 198.51.100.7", ["10.0.0.0/8"]) == "198.51.100.7"
    )


def test_invalid_trusted_proxy_entries_are_ignored():
    assert resolve_client_ip("10.0.0.2", "198.51.100.7", ["nonsense"]) == "10.0.0.2"


def test_missing_peer_is_anonymous():
    assert resolve_client_ip(None, None, []) == "anonymous"
