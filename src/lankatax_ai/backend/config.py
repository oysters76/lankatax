import logging

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from lankatax_ai.backend.types import PromptSource


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
    deepseek_api_key: SecretStr | None = None

    # Embedding cost (text-embedding-3-small: $0.02 per 1M tokens)
    embedding_cost_per_token: float = 0.02 / 1_000_000

    # Default chunking strategy (structural, semantic, recursive)
    chunk_strategy: str = "structural"

    # Logging
    log_level: str = "INFO"

    # Prompt loading. Defaults to versioned YAML files under prompts/.
    # Set LANKATAX_PROMPT_SOURCE=hub to pull from LangSmith Hub instead.
    prompt_source: PromptSource = Field(
        PromptSource.local, validation_alias=AliasChoices("LANKATAX_PROMPT_SOURCE")
    )
    prompt_hub_prefix: str | None = Field(
        None, validation_alias=AliasChoices("LANKATAX_PROMPT_HUB_PREFIX")
    )
    prompts_dir: str | None = Field(None, validation_alias=AliasChoices("LANKATAX_PROMPTS_DIR"))
    prompt_overrides: str | None = Field(
        None, validation_alias=AliasChoices("LANKATAX_PROMPT_OVERRIDES")
    )

    langchain_api_key: SecretStr = Field(
        validation_alias=AliasChoices("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")
    )
    langchain_project: str = Field(
        validation_alias=AliasChoices("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT")
    )
    langchain_tracing_v2: bool = Field(
        validation_alias=AliasChoices("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")
    )
    langchain_endpoint: str = Field(
        validation_alias=AliasChoices("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT")
    )

    model_config = SettingsConfigDict(env_file=".env")

    @property
    def connection_string(self) -> str:
        postgres_pwd = self.postgres_password.get_secret_value()
        return f"postgresql+asyncpg://{self.postgres_user}:{postgres_pwd}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db_name}"


settings = Settings()  # type: ignore[call-arg]  # required fields come from env/.env

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
