"""Build embedding policies from application settings."""

from __future__ import annotations

from lankatax_ai.backend.config import Settings, settings
from lankatax_ai.backend.ingestion.types import EmbeddingPolicy, RetryPolicy


def build_embedding_policy(config: Settings | None = None) -> EmbeddingPolicy:
    """Build the embedding policy from explicit settings or the process settings."""
    active = config or settings
    return EmbeddingPolicy(
        max_batch_tokens=active.embedding_batch_max_tokens,
        retry=RetryPolicy(
            max_retries=active.embedding_max_retries,
            backoff_base_seconds=active.embedding_backoff_base_seconds,
            backoff_max_seconds=active.embedding_backoff_max_seconds,
            jitter_ratio=active.embedding_jitter_ratio,
        ),
    )
