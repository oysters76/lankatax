import difflib
import enum
import json
from typing import Any
from uuid import UUID

import numpy as np

from lankatax_ai.backend.rag.types import RetrievalResult


class EnumEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, enum.Enum):
            return o.value
        return super().default(o)


def fuzzy_match(term: str, candidates: list[str], *, cutoff: float = 0.6) -> str | None:
    matches = difflib.get_close_matches(term, candidates, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def merge_result_sets(result_sets: list[list[RetrievalResult]]) -> list[RetrievalResult]:
    """Merge multiple retrieval result sets, deduplicate by chunk_id, keep highest score."""
    seen: dict[UUID, RetrievalResult] = {}
    for result_set in result_sets:
        for r in result_set:
            if r.chunk_id not in seen or r.score > seen[r.chunk_id].score:
                seen[r.chunk_id] = r
    return sorted(seen.values(), key=lambda r: r.score, reverse=True)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / norm)
