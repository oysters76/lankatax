from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Database (used by docker-compose and backend)
    postgres_db_name: str
    postgres_user: str
    postgres_password: SecretStr
    postgres_host: str
    postgres_port: int

    # LLM Providers
    openai_api_key: SecretStr
    google_api_key: SecretStr
    groq_api_key: SecretStr

    # Embedding cost (text-embedding-3-small: $0.02 per 1M tokens)
    embedding_cost_per_token: float = 0.02 / 1_000_000

    # Default chunking strategy (structural, semantic, recursive)
    chunk_strategy: str = "structural"

    # LangSmith Observability
    langchain_api_key: SecretStr
    langchain_project: str
    langchain_tracing_v2: bool
    langchain_endpoint: str

    model_config = SettingsConfigDict(env_file=".env")

    @property
    def connection_string(self) -> str:
        postgres_pwd = self.postgres_password.get_secret_value()
        return f"postgresql+asyncpg://{self.postgres_user}:{postgres_pwd}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db_name}"


settings = Settings()
