from __future__ import annotations

from typing import Any

import jwt
import pytest
from pydantic import SecretStr

from lankatax_ai.backend.config import Settings
from lankatax_ai.backend.jobs import client as client_module

pytestmark = pytest.mark.unit

TOKEN = jwt.encode(
    {
        "sub": "test",
        "server_url": "https://cloud.hatchet.run",
        "grpc_broadcast_address": "localhost:7070",
        "tenant_id": "00000000-0000-0000-0000-000000000000",
    },
    "x" * 32,
    algorithm="HS256",
)


class _FakeHatchet:
    def __init__(self, *, config: Any) -> None:
        self.config = config


def _settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "hatchet_client_token": SecretStr(TOKEN),
        "hatchet_client_tls_strategy": None,
        "hatchet_client_host_port": None,
    }
    return Settings(_env_file=None, **(defaults | overrides))


def test_builds_client_from_settings(monkeypatch):
    captured: list[_FakeHatchet] = []

    def fake_hatchet(*, config: Any) -> _FakeHatchet:
        built = _FakeHatchet(config=config)
        captured.append(built)
        return built

    monkeypatch.setattr(client_module, "Hatchet", fake_hatchet)

    built = client_module.build_hatchet_client(
        _settings(
            hatchet_client_tls_strategy="none",
            hatchet_client_host_port="selfhost:7077",
        )
    )

    assert built is captured[0]
    assert built.config.token == TOKEN
    assert built.config.host_port == "selfhost:7077"
    assert built.config.tls_config.strategy == "none"


def test_uses_sdk_defaults_when_optional_settings_are_absent(monkeypatch):
    monkeypatch.setattr(client_module, "Hatchet", _FakeHatchet)

    built = client_module.build_hatchet_client(_settings())

    assert built.config.token == TOKEN
    assert built.config.host_port == "localhost:7070"
    assert built.config.tls_config.strategy == "tls"


def test_missing_token_fails_fast(monkeypatch):
    monkeypatch.setattr(client_module, "Hatchet", _FakeHatchet)

    with pytest.raises(RuntimeError, match="HATCHET_CLIENT_TOKEN"):
        client_module.build_hatchet_client(_settings(hatchet_client_token=None))


def test_blank_token_fails_fast(monkeypatch):
    monkeypatch.setattr(client_module, "Hatchet", _FakeHatchet)

    with pytest.raises(RuntimeError, match="HATCHET_CLIENT_TOKEN"):
        client_module.build_hatchet_client(_settings(hatchet_client_token=SecretStr("   ")))
