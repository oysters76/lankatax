from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import pytest
from langsmith import run_helpers
from pydantic import SecretStr

from lankatax_ai.backend import tracing

pytestmark = pytest.mark.unit

CANONICAL_ENV = (
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGSMITH_ENDPOINT",
)


@pytest.fixture
def tracing_state(monkeypatch):
    """Known starting point: SDK present, tracing off, clean env."""
    for name in CANONICAL_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(tracing, "_SDK_AVAILABLE", True)
    monkeypatch.setattr(tracing, "_ACTIVE", False)
    monkeypatch.setattr(tracing.settings, "langchain_tracing_v2", False)
    monkeypatch.setattr(tracing.settings, "langchain_project", "lankatax-test")
    monkeypatch.setattr(tracing.settings, "langchain_endpoint", "https://api.smith.langchain.com")
    monkeypatch.setattr(tracing.settings, "langchain_api_key", SecretStr("test-key"))


def _enable(monkeypatch) -> None:
    monkeypatch.setattr(tracing.settings, "langchain_tracing_v2", True)


def test_disabled_tracing_returns_false_and_stays_quiet(tracing_state, caplog):
    with caplog.at_level(logging.DEBUG, logger=tracing.__name__):
        assert tracing.bootstrap_tracing() is False

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == []
    assert tracing.tracing_enabled() is False
    assert os.environ.get("LANGSMITH_TRACING") is None


def test_missing_sdk_warns_and_returns_false(tracing_state, monkeypatch, caplog):
    _enable(monkeypatch)
    monkeypatch.setattr(tracing, "_SDK_AVAILABLE", False)

    with caplog.at_level(logging.WARNING, logger=tracing.__name__):
        assert tracing.bootstrap_tracing() is False

    assert "langsmith package is missing" in caplog.text
    assert tracing.tracing_enabled() is False
    assert os.environ.get("LANGSMITH_TRACING") is None


def test_empty_api_key_warns_and_returns_false(tracing_state, monkeypatch, caplog):
    _enable(monkeypatch)
    monkeypatch.setattr(tracing.settings, "langchain_api_key", SecretStr("   "))

    with caplog.at_level(logging.WARNING, logger=tracing.__name__):
        assert tracing.bootstrap_tracing() is False

    assert "LANGSMITH_API_KEY is empty" in caplog.text
    assert tracing.tracing_enabled() is False
    assert os.environ.get("LANGSMITH_TRACING") is None


def test_enabled_tracing_exports_canonical_env(tracing_state, monkeypatch, caplog):
    _enable(monkeypatch)

    with caplog.at_level(logging.INFO, logger=tracing.__name__):
        assert tracing.bootstrap_tracing() is True

    assert tracing.tracing_enabled() is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "test-key"
    assert os.environ["LANGSMITH_PROJECT"] == "lankatax-test"
    assert os.environ["LANGSMITH_ENDPOINT"] == "https://api.smith.langchain.com"
    assert "test-key" not in caplog.text


def test_run_metadata_is_empty_when_inactive(tracing_state):
    assert tracing.run_metadata(query_type="factual", source="api", top_score=0.9) == {}


def test_run_metadata_keeps_only_supplied_fields(tracing_state, monkeypatch):
    monkeypatch.setattr(tracing, "_ACTIVE", True)

    metadata = tracing.run_metadata(
        query_type="calculation", mode="hybrid", source="cli", top_score=0.87
    )

    assert metadata == {
        "query_type": "calculation",
        "mode": "hybrid",
        "source": "cli",
        "top_score": 0.87,
    }


def test_run_metadata_includes_every_field(tracing_state, monkeypatch):
    monkeypatch.setattr(tracing, "_ACTIVE", True)

    metadata = tracing.run_metadata(
        query_type="factual",
        mode="vector",
        source="api",
        top_score=0.5,
        corpus_epoch="2026-11-21",
        config_fingerprint="cfg-abc123",
    )

    assert metadata == {
        "query_type": "factual",
        "mode": "vector",
        "source": "api",
        "top_score": 0.5,
        "corpus_epoch": "2026-11-21",
        "config_fingerprint": "cfg-abc123",
    }


def test_run_metadata_preserves_zero_and_empty_values(tracing_state, monkeypatch):
    """Falsy values are real values; only None should be dropped."""
    monkeypatch.setattr(tracing, "_ACTIVE", True)

    metadata = tracing.run_metadata(query_type="", top_score=0.0)

    assert metadata == {"query_type": "", "top_score": 0.0}


def test_run_id_is_none_when_inactive(tracing_state, monkeypatch):
    monkeypatch.setattr(tracing, "_ACTIVE", False)
    monkeypatch.setattr(tracing, "_SDK_AVAILABLE", True)

    assert tracing.run_id() is None


def test_run_id_returns_current_run(tracing_state, monkeypatch):
    monkeypatch.setattr(tracing, "_ACTIVE", True)
    monkeypatch.setattr(tracing, "_SDK_AVAILABLE", True)

    monkeypatch.setattr(run_helpers, "get_current_run_tree", lambda: SimpleNamespace(id="run-123"))

    assert tracing.run_id() == "run-123"


def test_run_id_is_none_without_an_active_run(tracing_state, monkeypatch):
    monkeypatch.setattr(tracing, "_ACTIVE", True)
    monkeypatch.setattr(tracing, "_SDK_AVAILABLE", True)

    monkeypatch.setattr(run_helpers, "get_current_run_tree", lambda: None)

    assert tracing.run_id() is None
