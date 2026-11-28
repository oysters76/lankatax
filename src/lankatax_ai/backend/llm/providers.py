from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from lankatax_ai.backend.config import settings
from lankatax_ai.backend.types import LlmProvider

DEFAULT_MODELS: dict[LlmProvider, str] = {
    LlmProvider.gemini: "gemini-2.5-flash",
    LlmProvider.groq: "llama-3.3-70b-versatile",
    LlmProvider.deepseek: "deepseek-chat",
    LlmProvider.openai: "gpt-4o-mini",
}

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def build_llm(
    provider: LlmProvider,
    *,
    model: str | None = None,
    max_tokens: int | None = None,
) -> BaseChatModel:
    """Build a chat model for one provider.

    SDKs are imported lazily so a missing optional dependency only affects the
    provider that needs it. Raises ``ValueError`` for an unsupported provider.
    """
    if provider is LlmProvider.gemini:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model or DEFAULT_MODELS[LlmProvider.gemini],
            google_api_key=settings.google_api_key.get_secret_value(),
            max_output_tokens=max_tokens,
        )

    if provider is LlmProvider.groq:
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model or DEFAULT_MODELS[LlmProvider.groq],
            api_key=settings.groq_api_key,
            max_tokens=max_tokens,
        )

    if provider is LlmProvider.deepseek:
        from langchain_openai import ChatOpenAI

        if settings.deepseek_api_key is None:
            raise ValueError("DEEPSEEK_API_KEY is not set")
        return ChatOpenAI(  # type: ignore[call-arg]  # max_tokens is a pydantic alias
            model=model or DEFAULT_MODELS[LlmProvider.deepseek],
            api_key=settings.deepseek_api_key,
            base_url=DEEPSEEK_BASE_URL,
            max_tokens=max_tokens,
        )

    if provider is LlmProvider.openai:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(  # type: ignore[call-arg]  # max_tokens is a pydantic alias
            model=model or DEFAULT_MODELS[LlmProvider.openai],
            api_key=settings.openai_api_key,
            max_tokens=max_tokens,
        )

    raise ValueError(f"Unsupported LLM provider: {provider!r}")
