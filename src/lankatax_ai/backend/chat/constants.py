"""Fixed chat data: refusal texts and prompt guidance."""

from __future__ import annotations

from lankatax_ai.backend.guardrails.types import GuardrailReason

__all__ = [
    "DEFAULT_REFUSAL",
    "HISTORY_CHAR_LIMIT",
    "REFUSAL_MESSAGES",
    "STRICT_SUFFIX",
]

REFUSAL_MESSAGES: dict[GuardrailReason, str] = {
    GuardrailReason.off_topic: (
        "I can only answer questions about Sri Lankan taxation and IRD procedures."
    ),
    GuardrailReason.injection: "I can't help with that request.",
    GuardrailReason.unsafe: "I can't help with that request.",
    GuardrailReason.pii: "Please remove personal identifiers and ask again.",
    GuardrailReason.empty: "Please ask a question about Sri Lankan tax.",
    GuardrailReason.too_long: "That question is too long. Please shorten it and try again.",
    GuardrailReason.insufficient_context: (
        "I couldn't find enough in the Sri Lankan tax documents to answer that. "
        "Try rephrasing, or ask about a specific tax type such as PAYE, VAT or income tax."
    ),
    GuardrailReason.ungrounded: (
        "I couldn't produce an answer that is fully supported by the available "
        "documents. Try a more specific question."
    ),
    GuardrailReason.rate_limited: "You're asking too quickly. Please wait a moment.",
}

DEFAULT_REFUSAL = "I can't answer that from the available tax documents."

STRICT_SUFFIX = (
    "\n\nOnly state facts that appear in the retrieved context and cite each one. "
    "If the context does not support an answer, say the documents do not cover it."
)

HISTORY_CHAR_LIMIT = 4000
