from __future__ import annotations

import pytest
from pydantic import ValidationError

from lankatax_ai.backend.config import Settings
from lankatax_ai.backend.types import PromptSource

pytestmark = pytest.mark.unit

LANGSMITH_VARS = (
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGSMITH_TRACING",
    "LANGSMITH_ENDPOINT",
)

HATCHET_VARS = (
    "HATCHET_CLIENT_TOKEN",
    "HATCHET_CLIENT_TLS_STRATEGY",
    "HATCHET_CLIENT_HOST_PORT",
    "HATCHET_DASHBOARD_URL",
    "HATCHET_WORKER_NAME",
    "HATCHET_WORKER_SLOTS",
)

JOB_VARS = (
    "JOB_REFRESH_CRON",
    "JOB_CACHE_EXPIRE_CRON",
    "JOB_EVAL_CRON",
    "JOB_SCRAPE_TIMEOUT_SECONDS",
    "JOB_INGEST_TIMEOUT_SECONDS",
    "JOB_CACHE_TIMEOUT_SECONDS",
    "JOB_EVAL_TIMEOUT_SECONDS",
    "JOB_SCHEDULE_TIMEOUT_SECONDS",
    "JOB_MAX_TOKENS_PER_RUN",
    "JOB_MAX_COST_USD",
    "JOB_SCRAPE_MAX_URLS",
    "JOB_INGEST_MAX_DOCUMENTS",
    "JOB_ENQUEUE_RATE_LIMIT_PER_MINUTE",
)


def _clear_langsmith(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in LANGSMITH_VARS:
        monkeypatch.delenv(name, raising=False)


def _clear_hatchet_and_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (*HATCHET_VARS, *JOB_VARS):
        monkeypatch.delenv(name, raising=False)


def _load() -> Settings:
    # Ignore the repo's .env so each test sees only the env it sets itself.
    return Settings(_env_file=None)


def test_scrape_base_host_is_read(monkeypatch):
    monkeypatch.setenv("SCRAPE_BASE_HOST", "example.lk")

    assert _load().scrape_base_host == "example.lk"


def test_ask_settings_are_read(monkeypatch):
    monkeypatch.setenv("ASK_TOP_K", "9")
    monkeypatch.setenv("ASK_GUARD_OUTPUT", "false")
    monkeypatch.setenv("ASK_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("ASK_RATE_LIMIT_PER_MINUTE", "3")
    monkeypatch.setenv("ASK_RATE_LIMIT_PER_DAY", "30")
    monkeypatch.setenv("ASK_GLOBAL_RATE_LIMIT_PER_MINUTE", "90")
    monkeypatch.setenv("ASK_GLOBAL_RATE_LIMIT_PER_DAY", "900")
    monkeypatch.setenv("ASK_MAX_CONCURRENCY", "3")
    monkeypatch.setenv("ASK_TRUSTED_PROXIES", "10.0.0.0/8")
    monkeypatch.setenv("ASK_MAX_BODY_BYTES", "4096")
    monkeypatch.setenv("ASK_DEDUP_THRESHOLD", "0.95")

    settings = _load()

    assert settings.ask_top_k == 9
    assert settings.ask_guard_output is False
    assert settings.ask_llm_provider.value == "gemini"
    assert settings.ask_rate_limit_per_minute == 3
    assert settings.ask_rate_limit_per_day == 30
    assert settings.ask_global_rate_limit_per_minute == 90
    assert settings.ask_global_rate_limit_per_day == 900
    assert settings.ask_max_concurrency == 3
    assert settings.ask_trusted_proxies == "10.0.0.0/8"
    assert settings.ask_max_body_bytes == 4096
    assert settings.ask_dedup_threshold == 0.95


def test_langsmith_names_are_read(monkeypatch):
    _clear_langsmith(monkeypatch)
    monkeypatch.setenv("LANGSMITH_API_KEY", "new-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "lankatax-dev")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://example.test")

    settings = _load()

    assert settings.langchain_api_key.get_secret_value() == "new-key"
    assert settings.langchain_project == "lankatax-dev"
    assert settings.langchain_tracing_v2 is True
    assert settings.langchain_endpoint == "https://example.test"


def test_legacy_langchain_names_still_work(monkeypatch):
    _clear_langsmith(monkeypatch)
    monkeypatch.setenv("LANGCHAIN_API_KEY", "legacy-key")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "lankatax")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_ENDPOINT", "https://legacy.test")

    settings = _load()

    assert settings.langchain_api_key.get_secret_value() == "legacy-key"
    assert settings.langchain_project == "lankatax"
    assert settings.langchain_tracing_v2 is True
    assert settings.langchain_endpoint == "https://legacy.test"


def test_langsmith_name_wins_when_both_are_set(monkeypatch):
    monkeypatch.setenv("LANGSMITH_PROJECT", "lankatax-dev")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "lankatax")

    assert _load().langchain_project == "lankatax-dev"


def test_missing_key_is_still_required(monkeypatch):
    _clear_langsmith(monkeypatch)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        _load()


def test_prompt_source_defaults_to_local():
    assert _load().prompt_source is PromptSource.local


def test_prompt_source_is_read_from_env(monkeypatch):
    monkeypatch.setenv("LANKATAX_PROMPT_SOURCE", "hub")

    assert _load().prompt_source is PromptSource.hub


def test_unknown_prompt_source_fails_at_startup(monkeypatch):
    monkeypatch.setenv("LANKATAX_PROMPT_SOURCE", "s3")

    with pytest.raises(ValidationError):
        _load()


def test_hatchet_settings_are_read(monkeypatch):
    _clear_hatchet_and_jobs(monkeypatch)
    monkeypatch.setenv("HATCHET_CLIENT_TOKEN", "hatchet-token")
    monkeypatch.setenv("HATCHET_CLIENT_TLS_STRATEGY", "none")
    monkeypatch.setenv("HATCHET_CLIENT_HOST_PORT", "localhost:7070")
    monkeypatch.setenv("HATCHET_DASHBOARD_URL", "https://cloud.hatchet.run")
    monkeypatch.setenv("HATCHET_WORKER_NAME", "lankatax-pipeline")
    monkeypatch.setenv("HATCHET_WORKER_SLOTS", "3")

    settings = _load()

    assert settings.hatchet_client_token is not None
    assert settings.hatchet_client_token.get_secret_value() == "hatchet-token"
    assert settings.hatchet_client_tls_strategy == "none"
    assert settings.hatchet_client_host_port == "localhost:7070"
    assert settings.hatchet_dashboard_url == "https://cloud.hatchet.run"
    assert settings.hatchet_worker_name == "lankatax-pipeline"
    assert settings.hatchet_worker_slots == 3


def test_job_policy_defaults(monkeypatch):
    _clear_hatchet_and_jobs(monkeypatch)

    settings = _load()

    assert settings.hatchet_client_token is None
    assert settings.hatchet_client_tls_strategy is None
    assert settings.hatchet_client_host_port is None
    assert settings.hatchet_worker_name == "lankatax-worker"
    assert settings.hatchet_worker_slots == 2
    assert settings.job_refresh_cron == "0 2 * * *"
    assert settings.job_cache_expire_cron == "0 3 * * *"
    assert settings.job_eval_cron == "0 4 * * 1"
    assert settings.job_scrape_timeout_seconds == 1800
    assert settings.job_ingest_timeout_seconds == 7200
    assert settings.job_cache_timeout_seconds == 300
    assert settings.job_eval_timeout_seconds == 3600
    assert settings.job_schedule_timeout_seconds == 3600
    assert settings.job_max_tokens_per_run == 1_000_000
    assert settings.job_max_cost_usd == 1.0
    assert settings.job_scrape_max_urls == 1000
    assert settings.job_ingest_max_documents == 500
    assert settings.job_enqueue_rate_limit_per_minute == 10


def test_job_policy_is_read_from_env(monkeypatch):
    _clear_hatchet_and_jobs(monkeypatch)
    monkeypatch.setenv("JOB_REFRESH_CRON", "15 1 * * *")
    monkeypatch.setenv("JOB_SCRAPE_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("JOB_MAX_COST_USD", "0.25")

    settings = _load()

    assert settings.job_refresh_cron == "15 1 * * *"
    assert settings.job_scrape_timeout_seconds == 60
    assert settings.job_max_cost_usd == 0.25


@pytest.mark.parametrize(
    ("name", "value"),
    [
        pytest.param("HATCHET_WORKER_SLOTS", "0", id="worker-slots-zero"),
        pytest.param("HATCHET_WORKER_SLOTS", "101", id="worker-slots-too-high"),
        pytest.param("JOB_SCRAPE_TIMEOUT_SECONDS", "0", id="scrape-timeout-zero"),
        pytest.param("JOB_INGEST_TIMEOUT_SECONDS", "0", id="ingest-timeout-zero"),
        pytest.param("JOB_CACHE_TIMEOUT_SECONDS", "0", id="cache-timeout-zero"),
        pytest.param("JOB_EVAL_TIMEOUT_SECONDS", "0", id="eval-timeout-zero"),
        pytest.param("JOB_SCHEDULE_TIMEOUT_SECONDS", "0", id="schedule-timeout-zero"),
        pytest.param("JOB_MAX_TOKENS_PER_RUN", "0", id="token-budget-zero"),
        pytest.param("JOB_MAX_COST_USD", "0", id="cost-budget-zero"),
        pytest.param("JOB_MAX_COST_USD", "-0.01", id="cost-budget-negative"),
        pytest.param("JOB_MAX_COST_USD", "inf", id="cost-budget-infinite"),
        pytest.param("JOB_SCRAPE_MAX_URLS", "0", id="scrape-max-urls-zero"),
        pytest.param("JOB_INGEST_MAX_DOCUMENTS", "0", id="ingest-max-documents-zero"),
        pytest.param(
            "JOB_ENQUEUE_RATE_LIMIT_PER_MINUTE",
            "0",
            id="enqueue-rate-limit-zero",
        ),
    ],
)
def test_job_policy_rejects_invalid_limits(monkeypatch, name, value):
    _clear_hatchet_and_jobs(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        _load()


def test_job_policy_accepts_boundary_limits(monkeypatch):
    _clear_hatchet_and_jobs(monkeypatch)
    monkeypatch.setenv("HATCHET_WORKER_SLOTS", "100")
    monkeypatch.setenv("JOB_SCRAPE_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("JOB_INGEST_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("JOB_CACHE_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("JOB_EVAL_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("JOB_SCHEDULE_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("JOB_MAX_TOKENS_PER_RUN", "1")
    monkeypatch.setenv("JOB_MAX_COST_USD", "0.01")
    monkeypatch.setenv("JOB_SCRAPE_MAX_URLS", "1")
    monkeypatch.setenv("JOB_INGEST_MAX_DOCUMENTS", "1")
    monkeypatch.setenv("JOB_ENQUEUE_RATE_LIMIT_PER_MINUTE", "1")

    settings = _load()

    assert settings.hatchet_worker_slots == 100
    assert settings.job_max_cost_usd == 0.01
    assert settings.job_max_tokens_per_run == 1


def test_unknown_hatchet_tls_strategy_fails_at_startup(monkeypatch):
    _clear_hatchet_and_jobs(monkeypatch)
    monkeypatch.setenv("HATCHET_CLIENT_TLS_STRATEGY", "bogus")

    with pytest.raises(ValidationError):
        _load()


def test_embedding_policy_defaults(monkeypatch):
    for name in (
        "EMBEDDING_BATCH_MAX_TOKENS",
        "EMBEDDING_MAX_RETRIES",
        "EMBEDDING_BACKOFF_BASE_SECONDS",
        "EMBEDDING_BACKOFF_MAX_SECONDS",
        "EMBEDDING_JITTER_RATIO",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = _load()

    assert settings.embedding_batch_max_tokens == 8000
    assert settings.embedding_max_retries == 3
    assert settings.embedding_backoff_base_seconds == 1.0
    assert settings.embedding_backoff_max_seconds == 10.0
    assert settings.embedding_jitter_ratio == 0.2


def test_embedding_policy_is_read_from_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BATCH_MAX_TOKENS", "4000")
    monkeypatch.setenv("EMBEDDING_MAX_RETRIES", "5")
    monkeypatch.setenv("EMBEDDING_BACKOFF_BASE_SECONDS", "0.5")
    monkeypatch.setenv("EMBEDDING_BACKOFF_MAX_SECONDS", "30")
    monkeypatch.setenv("EMBEDDING_JITTER_RATIO", "0.1")

    settings = _load()

    assert settings.embedding_batch_max_tokens == 4000
    assert settings.embedding_max_retries == 5
    assert settings.embedding_backoff_base_seconds == 0.5
    assert settings.embedding_backoff_max_seconds == 30
    assert settings.embedding_jitter_ratio == 0.1


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("EMBEDDING_BATCH_MAX_TOKENS", "0"),
        ("EMBEDDING_MAX_RETRIES", "-1"),
        ("EMBEDDING_JITTER_RATIO", "1.1"),
    ],
)
def test_embedding_policy_rejects_invalid_settings(monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        _load()
