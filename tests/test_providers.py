from __future__ import annotations

import subprocess
import sys
from typing import cast

import pytest
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from lankatax_ai.backend.llm import build_llm
from lankatax_ai.backend.llm.providers import DEFAULT_MODELS
from lankatax_ai.backend.types import LlmProvider

pytestmark = pytest.mark.unit


def test_gemini_uses_its_default_model():
    llm = build_llm(LlmProvider.gemini)

    assert isinstance(llm, ChatGoogleGenerativeAI)
    assert llm.model == DEFAULT_MODELS[LlmProvider.gemini]


def test_groq_uses_its_default_model():
    llm = build_llm(LlmProvider.groq)

    assert isinstance(llm, ChatGroq)
    assert llm.model_name == DEFAULT_MODELS[LlmProvider.groq]


def test_model_override_is_respected():
    llm = build_llm(LlmProvider.groq, model="llama-3.1-8b-instant")

    assert llm.model_name == "llama-3.1-8b-instant"


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        build_llm(cast(LlmProvider, "openai"))


def test_importing_providers_does_not_pull_in_cli_layer():
    code = (
        "import sys\n"
        "import lankatax_ai.backend.llm.providers\n"
        "leaked = [m for m in sys.modules if m.startswith('lankatax_ai.cli')]\n"
        "assert not leaked, leaked\n"
    )

    subprocess.run([sys.executable, "-c", code], check=True)
