import difflib
import enum
import json

import numpy as np


class EnumEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, enum.Enum):
            return o.value
        return super().default(o)


def fuzzy_match(term: str, candidates: list[str], *, cutoff: float = 0.6) -> str | None:
    matches = difflib.get_close_matches(term, candidates, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / norm)
