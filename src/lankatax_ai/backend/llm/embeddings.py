"""Embedding model construction for transports and orchestrators.

Services receive the resulting client; they never build one themselves.
"""

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from lankatax_ai.backend.config import settings

EMBEDDING_MODEL = "text-embedding-3-small"


def build_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=settings.openai_api_key,
    )
