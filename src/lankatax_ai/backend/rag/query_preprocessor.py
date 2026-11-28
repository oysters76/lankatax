from __future__ import annotations

import importlib.resources
import json
import string

from lankatax_ai.backend.rag.types import PreprocessedQuery
from lankatax_ai.backend.types import QueryType
from lankatax_ai.backend.utils import fuzzy_match


class QueryPreprocessor:
    def __init__(self, fuzzy_threshold: float = 0.75) -> None:
        data_dir = importlib.resources.files("lankatax_ai.backend.rag.data")
        self._acronyms: dict[str, str] = json.loads((data_dir / "tax_acronyms.json").read_text())
        self._terms: list[str] = json.loads((data_dir / "tax_terms.json").read_text())
        self._fuzzy_threshold = fuzzy_threshold
        self._term_words: set[str] = {word for phrase in self._terms for word in phrase.split()}
        signals: dict[str, list[str]] = json.loads((data_dir / "query_signals.json").read_text())
        self._calculation_signals: list[str] = signals["calculation"]
        self._procedural_signals: list[str] = signals["procedural"]

    def process(self, query: str) -> PreprocessedQuery:
        corrected = self._correct_spelling(query)
        expanded = self._expand_acronyms(corrected)
        query_type = self._classify(expanded)
        return PreprocessedQuery(
            original=query,
            expanded=expanded,
            query_type=query_type,
        )

    def _expand_acronyms(self, query: str) -> str:
        expanded = []
        for token in query.split():
            stripped = token.strip(string.punctuation)
            clean = stripped.upper()
            # Only uppercase tokens can be acronyms: lowercase words such as
            # "it" collide with IT (Income Tax) and must not be expanded.
            if not stripped or not stripped.isupper() or len(clean) < 3:
                expanded.append(token)
                continue

            key = clean if clean in self._acronyms else None
            if key is None:
                key = fuzzy_match(clean, list(self._acronyms.keys()), cutoff=self._fuzzy_threshold)
            if key is None:
                expanded.append(token)
            else:
                expanded.append(self._insert(token, stripped, f"{key} {self._acronyms[key]}"))
        return " ".join(expanded)

    @staticmethod
    def _insert(token: str, stripped: str, replacement: str) -> str:
        """Put a replacement where the token's word is, keeping punctuation."""
        start = token.find(stripped)
        if start < 0:
            return f"{token} {replacement}"
        return f"{token[:start]}{replacement}{token[start + len(stripped) :]}"

    def _correct_spelling(self, query: str) -> str:
        tokens = query.split()
        corrected = []
        for token in tokens:
            stripped = token.strip(string.punctuation)
            lower = stripped.lower()
            if (
                not stripped.isalpha()
                or stripped.upper() in self._acronyms
                or lower in self._term_words
            ):
                corrected.append(token)
                continue
            match = fuzzy_match(lower, list(self._term_words), cutoff=0.9)
            corrected.append(match if match else token)
        return " ".join(corrected)

    def _classify(self, query: str) -> QueryType:
        query_lower = query.lower()
        if any(signal in query_lower for signal in self._calculation_signals):
            return QueryType.calculation
        if any(signal in query_lower for signal in self._procedural_signals):
            return QueryType.procedural
        return QueryType.factual
