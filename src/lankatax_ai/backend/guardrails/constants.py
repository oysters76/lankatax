"""Fixed guardrail data: patterns, category mapping and answer regexes."""

from __future__ import annotations

import re

from lankatax_ai.backend.guardrails.types import GuardrailReason

__all__ = [
    "CATEGORY_REASONS",
    "CITATION_RE",
    "DANGEROUS_HTML_RE",
    "INJECTION_PATTERNS",
    "PII_PATTERNS",
]

INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(all\s+|any\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(
        r"disregard\s+(the\s+)?(system|previous|above)\s+(prompt|instructions)", re.IGNORECASE
    ),
    re.compile(r"reveal\s+(your\s+)?(system\s+prompt|hidden\s+instructions)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an|the)\s+", re.IGNORECASE),
    re.compile(r"pretend\s+(to\s+be|you\s+are)\s+", re.IGNORECASE),
)

PII_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b\d{9}[vVxX]\b"), "[redacted-nic]"),
    (re.compile(r"\b\d{12}\b"), "[redacted-nic]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[redacted-email]"),
    (re.compile(r"\b(?:\+94|0)\d{9}\b"), "[redacted-phone]"),
)

CATEGORY_REASONS: dict[str, GuardrailReason] = {
    "off_topic": GuardrailReason.off_topic,
    "injection": GuardrailReason.injection,
    "unsafe": GuardrailReason.unsafe,
    "pii": GuardrailReason.pii,
}

CITATION_RE = re.compile(r"\[(\d+)\]")
DANGEROUS_HTML_RE = re.compile(r"<\s*(script|iframe|object|embed|style)\b", re.IGNORECASE)
