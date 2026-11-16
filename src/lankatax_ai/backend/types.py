from enum import Enum


class DocumentKind(Enum):
    pdf = "pdf"
    page = "page"


class LlmProvider(Enum):
    gemini = "gemini"
    groq = "groq"


MODEL_TOKEN_BUDGETS: dict[LlmProvider, int] = {
    LlmProvider.gemini: 4000,
    LlmProvider.groq: 2500,
}
