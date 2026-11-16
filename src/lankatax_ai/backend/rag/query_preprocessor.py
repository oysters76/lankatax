from __future__ import annotations

import importlib.resources
import json

from lankatax_ai.backend.rag.types import PreprocessedQuery, QueryType
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
        tokens = query.split()
        expanded = []
        for token in tokens:
            clean = token.upper()
            if clean in self._acronyms:
                expanded.append(f"{token} {self._acronyms[clean]}")
            elif clean.isupper() and len(clean) >= 3:
                match = fuzzy_match(
                    clean, list(self._acronyms.keys()), cutoff=self._fuzzy_threshold
                )
                if match:
                    expanded.append(f"{match} {self._acronyms[match]}")
                else:
                    expanded.append(token)
            else:
                expanded.append(token)
        return " ".join(expanded)

    def _correct_spelling(self, query: str) -> str:
        tokens = query.split()
        corrected = []
        for token in tokens:
            token_lower = token.lower()
            if token_lower in self._term_words:
                corrected.append(token)
            else:
                match = fuzzy_match(token_lower, list(self._term_words), cutoff=0.7)
                if match:
                    corrected.append(match)
                else:
                    corrected.append(token)
        return " ".join(corrected)

    def _classify(self, query: str) -> QueryType:
        query_lower = query.lower()
        if any(signal in query_lower for signal in self._calculation_signals):
            return QueryType.calculation
        if any(signal in query_lower for signal in self._procedural_signals):
            return QueryType.procedural
        return QueryType.factual
