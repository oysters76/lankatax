from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from lankatax_ai.backend.config import settings
from lankatax_ai.backend.types import LlmProvider

DEFAULT_MODELS: dict[LlmProvider, str] = {
    LlmProvider.gemini: "gemini-2.5-flash",
    LlmProvider.groq: "llama-3.3-70b-versatile",
}


def build_llm(provider: LlmProvider, *, model: str | None = None) -> BaseChatModel:
    """Build a chat model for one provider.

    SDKs are imported lazily so a missing optional dependency only affects the
    provider that needs it. Raises ``ValueError`` for an unsupported provider.
    """
    if provider is LlmProvider.gemini:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model or DEFAULT_MODELS[LlmProvider.gemini],
            google_api_key=settings.google_api_key.get_secret_value(),
        )

    if provider is LlmProvider.groq:
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model or DEFAULT_MODELS[LlmProvider.groq],
            api_key=settings.groq_api_key.get_secret_value(),
        )

    raise ValueError(f"Unsupported LLM provider: {provider!r}")
