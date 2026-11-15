from __future__ import annotations

from typing import Self

LLM_PROVIDERS = frozenset({"gemini", "groq"})

MODEL_TOKEN_BUDGETS: dict[str, int] = {
    "gemini": 4000,
    "groq": 2500,
}


class CLIState:
    _instance: CLIState | None = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._llm_provider = "gemini"
            cls._token_budget: int | None = None
            cls._rerank_threshold = 3
            cls._vector_weight = 0.7
            cls._dedup_threshold = 0.6
            cls._top_k = 5
        return cls._instance

    @property
    def llm_provider(self) -> str:
        return self._llm_provider

    @llm_provider.setter
    def llm_provider(self, value: str) -> None:
        if value not in LLM_PROVIDERS:
            raise ValueError(
                f"Unknown LLM provider: {value!r}. Choose from {sorted(LLM_PROVIDERS)}"
            )
        self._llm_provider = value

    @property
    def token_budget(self) -> int:
        if self._token_budget is not None:
            return self._token_budget
        return MODEL_TOKEN_BUDGETS[self._llm_provider]

    @token_budget.setter
    def token_budget(self, value: int) -> None:
        if value < 500 or value > 16000:
            raise ValueError(f"Token budget must be 500-16000, got {value}")
        self._token_budget = value

    @property
    def rerank_threshold(self) -> int:
        return self._rerank_threshold

    @rerank_threshold.setter
    def rerank_threshold(self, value: int) -> None:
        if value < 1 or value > 10:
            raise ValueError(f"Rerank threshold must be 1-10, got {value}")
        self._rerank_threshold = value

    @property
    def vector_weight(self) -> float:
        return self._vector_weight

    @vector_weight.setter
    def vector_weight(self, value: float) -> None:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Vector weight must be 0.0-1.0, got {value}")
        self._vector_weight = value

    @property
    def dedup_threshold(self) -> float:
        return self._dedup_threshold

    @dedup_threshold.setter
    def dedup_threshold(self, value: float) -> None:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Dedup threshold must be 0.0-1.0, got {value}")
        self._dedup_threshold = value

    @property
    def top_k(self) -> int:
        return self._top_k

    @top_k.setter
    def top_k(self, value: int) -> None:
        if value < 1 or value > 50:
            raise ValueError(f"Top-k must be 1-50, got {value}")
        self._top_k = value


cli_state = CLIState()
