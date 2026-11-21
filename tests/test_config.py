from __future__ import annotations

import pytest
from pydantic import ValidationError

from lankatax_ai.backend.config import Settings
from lankatax_ai.backend.types import PromptSource

pytestmark = pytest.mark.unit

LANGSMITH_VARS = (
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGSMITH_TRACING",
    "LANGSMITH_ENDPOINT",
)


def _clear_langsmith(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in LANGSMITH_VARS:
        monkeypatch.delenv(name, raising=False)


def _load() -> Settings:
    # Ignore the repo's .env so each test sees only the env it sets itself.
    return Settings(_env_file=None)


def test_langsmith_names_are_read(monkeypatch):
    _clear_langsmith(monkeypatch)
    monkeypatch.setenv("LANGSMITH_API_KEY", "new-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "lankatax-dev")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://example.test")

    settings = _load()

    assert settings.langchain_api_key.get_secret_value() == "new-key"
    assert settings.langchain_project == "lankatax-dev"
    assert settings.langchain_tracing_v2 is True
    assert settings.langchain_endpoint == "https://example.test"


def test_legacy_langchain_names_still_work(monkeypatch):
    _clear_langsmith(monkeypatch)
    monkeypatch.setenv("LANGCHAIN_API_KEY", "legacy-key")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "lankatax")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_ENDPOINT", "https://legacy.test")

    settings = _load()

    assert settings.langchain_api_key.get_secret_value() == "legacy-key"
    assert settings.langchain_project == "lankatax"
    assert settings.langchain_tracing_v2 is True
    assert settings.langchain_endpoint == "https://legacy.test"


def test_langsmith_name_wins_when_both_are_set(monkeypatch):
    monkeypatch.setenv("LANGSMITH_PROJECT", "lankatax-dev")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "lankatax")

    assert _load().langchain_project == "lankatax-dev"


def test_missing_key_is_still_required(monkeypatch):
    _clear_langsmith(monkeypatch)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        _load()


def test_prompt_source_defaults_to_local():
    assert _load().prompt_source is PromptSource.local


def test_prompt_source_is_read_from_env(monkeypatch):
    monkeypatch.setenv("LANKATAX_PROMPT_SOURCE", "hub")

    assert _load().prompt_source is PromptSource.hub


def test_unknown_prompt_source_fails_at_startup(monkeypatch):
    monkeypatch.setenv("LANKATAX_PROMPT_SOURCE", "s3")

    with pytest.raises(ValidationError):
        _load()
