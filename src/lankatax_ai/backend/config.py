import logging
from typing import Literal

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

    # API authentication. Optional here so the CLI, ingest and tests keep
    # working without them; the API validates them in create_app().
    api_auth_username: str | None = None
    api_auth_password_hash: SecretStr | None = None
    api_auth_secret: SecretStr | None = Field(
        None, validation_alias=AliasChoices("API_AUTH_SECRET", "API_KEY_TOKEN")
    )
    api_auth_token_expire_minutes: int = 30
    api_auth_algorithm: str = "HS256"

    # True = /docs, /redoc and /openapi.json require the bearer token.
    api_docs_require_auth: bool = True

    # Hatchet background jobs. The token is optional here so the CLI, ingest
    # and tests keep working without Hatchet; job wiring validates it later.
    hatchet_client_token: SecretStr | None = None
    hatchet_client_tls_strategy: Literal["tls", "none"] | None = None
    hatchet_dashboard_url: str | None = None
    hatchet_worker_name: str = "lankatax-worker"
    hatchet_worker_slots: int = Field(2, ge=1, le=100)

    # Scheduled jobs. Expressions are UTC and evaluated by Hatchet.
    job_refresh_cron: str = "0 2 * * *"
    job_cache_expire_cron: str = "0 3 * * *"
    job_eval_cron: str = "0 4 * * 1"

    # Per-job execution limits. Hatchet defaults are much shorter than a crawl.
    job_scrape_timeout_seconds: int = Field(1800, gt=0)
    job_ingest_timeout_seconds: int = Field(7200, gt=0)
    job_cache_timeout_seconds: int = Field(300, gt=0)
    job_eval_timeout_seconds: int = Field(3600, gt=0)
    job_schedule_timeout_seconds: int = Field(3600, gt=0)

    # Guardrails for a single run.
    job_max_tokens_per_run: int = Field(1_000_000, gt=0)
    job_max_cost_usd: float = Field(1.0, gt=0, allow_inf_nan=False)
    job_scrape_max_urls: int = Field(1000, gt=0)
    job_ingest_max_documents: int = Field(500, gt=0)
    job_enqueue_rate_limit_per_minute: int = Field(10, gt=0)

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
