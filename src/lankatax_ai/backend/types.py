from enum import Enum


class DocumentKind(Enum):
    pdf = "pdf"
    page = "page"


class ChunkStrategy(Enum):
    structural = "structural"
    semantic = "semantic"
    recursive = "recursive"


class SearchMode(Enum):
    hybrid = "hybrid"
    vector = "vector"
    keyword = "keyword"


class SearchSource(Enum):
    cli = "cli"
    api = "api"


class QueryType(Enum):
    factual = "factual"
    calculation = "calculation"
    procedural = "procedural"


class LlmProvider(Enum):
    gemini = "gemini"
    groq = "groq"
    deepseek = "deepseek"
    openai = "openai"


class PromptSource(Enum):
    local = "local"
    hub = "hub"


MODEL_TOKEN_BUDGETS: dict[LlmProvider, int] = {
    LlmProvider.gemini: 4000,
    LlmProvider.groq: 2500,
    LlmProvider.deepseek: 6000,
    LlmProvider.openai: 8000,
}
