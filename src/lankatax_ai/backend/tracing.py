from __future__ import annotations

import logging
import os

from lankatax_ai.backend.config import settings

logger = logging.getLogger(__name__)

try:
    import langsmith  # noqa: F401
except ImportError:
    _SDK_AVAILABLE = False
else:
    _SDK_AVAILABLE = True

_ACTIVE = False


def bootstrap_tracing() -> bool:
    """Export LangSmith settings and report whether tracing is active.

    Safe when tracing is off or the SDK is missing: returns ``False`` instead
    of raising. Call once at startup, before any LLM client is created.
    """
    global _ACTIVE

    if not settings.langchain_tracing_v2:
        logger.debug("LangSmith tracing disabled")
        _ACTIVE = False
        return False

    if not _SDK_AVAILABLE:
        logger.warning(
            "LangSmith tracing is enabled but the langsmith package is missing; "
            "continuing without traces. Install it or unset LANGSMITH_TRACING."
        )
        _ACTIVE = False
        return False

    api_key = settings.langchain_api_key.get_secret_value()
    if not api_key.strip():
        logger.warning(
            "LangSmith tracing is enabled but LANGSMITH_API_KEY is empty; "
            "continuing without traces."
        )
        _ACTIVE = False
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langchain_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langchain_endpoint
    _ACTIVE = True
    logger.info("LangSmith tracing enabled (project=%s)", settings.langchain_project)
    return True


def tracing_enabled() -> bool:
    """Whether bootstrap_tracing() successfully activated tracing."""
    return _ACTIVE


def run_metadata(
    *,
    query_type: str | None = None,
    mode: str | None = None,
    source: str | None = None,
    top_score: float | None = None,
    corpus_epoch: str | None = None,
    config_fingerprint: str | None = None,
) -> dict[str, str | float]:
    """Metadata for the current trace; empty when tracing is inactive.

    Pass serialized values (``Enum.value``, not the enum) so the dict stays
    JSON-friendly for LangSmith.
    """
    if not _ACTIVE:
        return {}

    values: dict[str, str | float | None] = {
        "query_type": query_type,
        "mode": mode,
        "source": source,
        "top_score": top_score,
        "corpus_epoch": corpus_epoch,
        "config_fingerprint": config_fingerprint,
    }
    metadata: dict[str, str | float] = {}
    for key, value in values.items():
        if value is not None:
            metadata[key] = value
    return metadata


def run_id() -> str | None:
    """ID of the active LangSmith run, or None when tracing is off/absent."""
    if not _ACTIVE or not _SDK_AVAILABLE:
        return None

    try:
        from langsmith.run_helpers import get_current_run_tree
    except ImportError:
        return None

    run = get_current_run_tree()
    return str(run.id) if run is not None else None
