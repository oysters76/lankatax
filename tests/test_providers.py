from __future__ import annotations

import subprocess
import sys
from typing import cast

import pytest
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from lankatax_ai.backend.config import settings
from lankatax_ai.backend.llm import build_llm
from lankatax_ai.backend.llm.providers import DEFAULT_MODELS
from lankatax_ai.backend.types import MODEL_TOKEN_BUDGETS, LlmProvider

pytestmark = pytest.mark.unit

PROVIDER_CASES = [
    (LlmProvider.gemini, ChatGoogleGenerativeAI, "model"),
    (LlmProvider.groq, ChatGroq, "model_name"),
    (LlmProvider.deepseek, ChatOpenAI, "model_name"),
    (LlmProvider.openai, ChatOpenAI, "model_name"),
]


@pytest.mark.parametrize(("provider", "expected_type", "model_attr"), PROVIDER_CASES)
def test_each_provider_uses_its_default_model(provider, expected_type, model_attr):
    llm = build_llm(provider)

    assert isinstance(llm, expected_type)
    assert getattr(llm, model_attr) == DEFAULT_MODELS[provider]


def test_model_override_is_respected():
    llm = build_llm(LlmProvider.groq, model="llama-3.1-8b-instant")

    assert llm.model_name == "llama-3.1-8b-instant"


@pytest.mark.parametrize(
    ("provider", "attribute"),
    [
        (LlmProvider.gemini, "max_output_tokens"),
        (LlmProvider.groq, "max_tokens"),
        (LlmProvider.deepseek, "max_tokens"),
        (LlmProvider.openai, "max_tokens"),
    ],
)
def test_max_tokens_override_is_respected(provider, attribute):
    llm = build_llm(provider, max_tokens=321)

    assert getattr(llm, attribute) == 321


def test_deepseek_points_at_deepseek_base_url():
    llm = build_llm(LlmProvider.deepseek)

    assert isinstance(llm, ChatOpenAI)
    assert llm.openai_api_base == "https://api.deepseek.com"


def test_deepseek_without_a_key_raises(monkeypatch):
    monkeypatch.setattr(settings, "deepseek_api_key", None)

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        build_llm(LlmProvider.deepseek)


def test_every_provider_has_a_model_and_a_token_budget():
    assert set(DEFAULT_MODELS) == set(LlmProvider)
    assert set(MODEL_TOKEN_BUDGETS) == set(LlmProvider)


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        build_llm(cast(LlmProvider, "openrouter"))


def test_importing_providers_does_not_pull_in_cli_layer():
    code = (
        "import sys\n"
        "import lankatax_ai.backend.llm.providers\n"
        "leaked = [m for m in sys.modules if m.startswith('lankatax_ai.cli')]\n"
        "assert not leaked, leaked\n"
    )

    subprocess.run([sys.executable, "-c", code], check=True)
