"""API adapter tests.

Every test runs the real routers and the real service functions; only the outer
adapters (retriever, cache, analytics, stats, database session) are fakes. No
database, network or ``.env`` is involved.
"""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import jwt
import pytest
from fastapi.testclient import TestClient
from pwdlib import PasswordHash
from pydantic import SecretStr

from lankatax_ai.backend.analytics.types import StatsSnapshot
from lankatax_ai.backend.api import create_app
from lankatax_ai.backend.api.deps import ApiDeps
from lankatax_ai.backend.api.security import AuthConfig
from lankatax_ai.backend.cache.types import CacheStats
from lankatax_ai.backend.config import Settings
from lankatax_ai.backend.documents import ScrapeStatusRecord
from lankatax_ai.backend.ingestion.types import EmbeddingPolicy, IngestionStatus
from lankatax_ai.backend.rag.types import RetrievalResult
from lankatax_ai.backend.services.cache import CacheDeps
from lankatax_ai.backend.services.errors import NotFoundError
from lankatax_ai.backend.services.ingest import IngestDeps
from lankatax_ai.backend.services.jobs import (
    JobDeps,
    JobDetails,
    JobFilter,
    JobReceipt,
    JobRequest,
    JobStatus,
    JobSummary,
)
from lankatax_ai.backend.services.scrape import ScrapeDeps
from lankatax_ai.backend.services.search import SearchDeps
from lankatax_ai.backend.services.stats import StatsDeps
from lankatax_ai.backend.types import SearchMode, SearchSource

pytestmark = pytest.mark.unit

USERNAME = "admin"
PASSWORD = "secret"
SECRET = "test-secret-key-for-the-lankatax-api-0123456789"
TOKEN_URL = "/api/v1/auth/token"
JOBS_URL = "/api/v1/jobs"
SCRAPE_URL = "/api/v1/scrape"
INGEST_URL = "/api/v1/ingest"
PIPELINE_REFRESH_URL = "/api/v1/pipeline/refresh"

AUTH = AuthConfig(
    username=USERNAME,
    password_hash=PasswordHash.recommended().hash(PASSWORD),
    secret=SECRET,
    expire_minutes=30,
)

EXPECTED_ACTIONS = {
    "token",
    "ask",
    "search",
    "stats",
    "cache_stats",
    "cache_clear",
    "cache_expire",
    "ingest",
    "ingest_status",
    "scrape",
    "scrape_urls",
    "scrape_status",
    "pipeline_refresh",
    "jobs",
    "health",
    "docs",
}

#: Job, scrape, ingest and refresh routes; all of them require a bearer token.
#: They replaced the 501 stubs, so the stub test became this missing-token check.
PROTECTED_ROUTES = [
    ("POST", INGEST_URL),
    ("GET", f"{INGEST_URL}/status"),
    ("GET", f"{SCRAPE_URL}/urls"),
    ("POST", SCRAPE_URL),
    ("GET", f"{SCRAPE_URL}/status"),
    ("POST", PIPELINE_REFRESH_URL),
    ("GET", JOBS_URL),
    ("POST", JOBS_URL),
    ("GET", f"{JOBS_URL}/run-1"),
    ("POST", f"{JOBS_URL}/run-1/cancel"),
    ("POST", f"{JOBS_URL}/run-1/replay"),
]


class _FakeSession:
    async def execute(self, statement: Any) -> None:
        return None

    async def commit(self) -> None:
        return None


class _BrokenSessionFactory:
    """A session factory that fails on connection, for the health test."""

    def __call__(self) -> Any:
        raise RuntimeError("database is down")


@asynccontextmanager
async def _session_factory() -> AsyncIterator[_FakeSession]:
    yield _FakeSession()


def _result() -> RetrievalResult:
    return RetrievalResult(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content="PAYE is withheld by the employer.",
        score=0.91,
        vector_score=0.88,
        keyword_score=0.72,
        metadata={
            "from_url": "https://www.ird.gov.lk/paye",
            "document_title": "PAYE guide",
            "document_kind": "page",
            "order": 0,
        },
        embedding=[0.1, 0.2, 0.3],
    )


class _FakeEmbeddings:
    async def aembed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class _FakeRetriever:
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error

    async def search(self, query, *, mode, top_k, filters, include_embeddings):
        if self._error is not None:
            raise self._error
        return [_result()][:top_k]


class _FakeExpander:
    async def expand(self, query: str) -> list[str]:
        return [f"{query} (expanded)"]


class _FakeReranker:
    async def rerank(self, query, results, *, top_k):
        return results[:top_k]


class _FakeCache:
    async def get(self, query_text, query_embedding):
        return None

    async def put(self, query_text, query_embedding, results) -> None:
        return None


class _FakeAnalytics:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def log_search(self, **event: Any) -> None:
        self.events.append(event)


class _FakeStatsReader:
    async def snapshot(self, *, threshold: float) -> StatsSnapshot:
        return StatsSnapshot(
            total_queries=4,
            avg_latency_ms=12.5,
            avg_top_score=0.8,
            cache_hit_count=1,
            multi_query_count=2,
            hit_count=3,
            no_result_queries=["unmatched query"],
        )


class _FakeCacheAdmin:
    async def get_stats(self) -> CacheStats:
        return CacheStats(
            total_entries=7,
            total_hits=21,
            hit_rate=0.75,
            oldest_entry_hours=5.5,
        )

    async def clear(self) -> int:
        return 7

    async def invalidate_expired(self) -> int:
        return 2


class _FakeRegistry:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def get_all(self) -> list:
        return []


class _FakeScrapeStore:
    async def scrape_status(self) -> ScrapeStatusRecord:
        now = datetime.now(UTC)
        return ScrapeStatusRecord(total=2, recent=1, first_scrape=now, last_scrape=now)


class _FakeIngestRepository:
    async def status(self) -> IngestionStatus:
        return IngestionStatus(total_documents=3, pending=1, ingested=2)


async def _fake_finalizer(run_id: str | None = None) -> int:
    return 1


class _FakeJobGateway:
    async def enqueue(self, request: JobRequest) -> JobReceipt:
        return JobReceipt(run_id="run-1")

    async def list_jobs(self, filters: JobFilter) -> list[JobSummary]:
        return []

    async def get(self, run_id: str) -> JobDetails:
        raise NotFoundError("job not found")

    async def status(self, run_id: str) -> JobStatus:
        return JobStatus.queued

    async def cancel(self, run_id: str) -> None:
        return None

    async def replay(self, run_id: str) -> None:
        return None


@dataclasses.dataclass(frozen=True, slots=True)
class _Harness:
    deps: ApiDeps
    analytics: _FakeAnalytics


def _harness(
    *,
    session_factory: Any = _session_factory,
    retriever_error: Exception | None = None,
) -> _Harness:
    analytics = _FakeAnalytics()
    deps = ApiDeps(
        session_factory=session_factory,
        search=SearchDeps(
            embeddings=_FakeEmbeddings(),
            retriever_factory=lambda session, options: _FakeRetriever(retriever_error),
            expander_factory=lambda provider: _FakeExpander(),
            reranker_factory=lambda provider, threshold: _FakeReranker(),
            cache_factory=lambda session: _FakeCache(),
            analytics_factory=lambda session: analytics,
        ),
        stats=StatsDeps(
            reader_factory=lambda session: _FakeStatsReader(),
            no_result_threshold=1.0,
        ),
        cache=CacheDeps(admin_factory=lambda session: _FakeCacheAdmin()),
        jobs=JobDeps(gateway=_FakeJobGateway()),
        scrape=ScrapeDeps(
            registry=_FakeRegistry(),  # type: ignore[arg-type]
            crawler_factory=lambda: None,  # type: ignore[arg-type,return-value]
            store_factory=lambda _session: _FakeScrapeStore(),  # type: ignore[arg-type]
        ),
        ingest=IngestDeps(
            repository_factory=lambda _session: _FakeIngestRepository(),  # type: ignore[arg-type]
            embeddings=_FakeEmbeddings(),
            embedding_policy=EmbeddingPolicy(),
            corpus_finalizer=_fake_finalizer,
        ),
        auth=AUTH,
    )
    return _Harness(deps=deps, analytics=analytics)


@pytest.fixture
def harness() -> _Harness:
    return _harness()


@pytest.fixture
def client(harness: _Harness) -> TestClient:
    return TestClient(create_app(api_deps=harness.deps, auth_config=AUTH, docs_require_auth=True))


@pytest.fixture
def open_client() -> TestClient:
    return TestClient(
        create_app(api_deps=_harness().deps, auth_config=AUTH, docs_require_auth=False)
    )


def _login(client: TestClient) -> str:
    response = client.post(TOKEN_URL, data={"username": USERNAME, "password": PASSWORD})
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _mint(**claims: Any) -> str:
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": USERNAME,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=5)).timestamp()),
        "typ": "access",
    }
    payload.update(claims)
    return jwt.encode(payload, SECRET, algorithm="HS256")


def _follow(client: TestClient, target: dict[str, Any], token: str | None = None) -> int:
    headers = _auth(token) if token else {}
    response = client.request(target["method"], target["href"], headers=headers)
    return response.status_code


def test_service_root_links_resolve(client: TestClient) -> None:
    body = client.get("/").json()

    assert set(body["links"]) == {"api_root", "health", "docs", "openapi"}
    for target in body["links"].values():
        assert _follow(client, target) not in (404, 405)


def test_health_reports_the_database(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert set(response.json()["links"]) == {"self", "api_root"}


def test_health_reports_an_outage() -> None:
    harness = _harness(session_factory=_BrokenSessionFactory())
    client = TestClient(create_app(api_deps=harness.deps, auth_config=AUTH))

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["code"] == "unavailable"
    assert response.json()["links"]["api_root"]["href"].endswith("/api/v1")


def test_api_root_advertises_every_action(client: TestClient) -> None:
    body = client.get("/api/v1").json()

    assert set(body["links"]) == EXPECTED_ACTIONS
    for target in body["links"].values():
        assert _follow(client, target) not in (404, 405)


def test_api_root_marks_stubs_and_guarded_links(client: TestClient) -> None:
    links = client.get("/api/v1").json()["links"]

    assert links["ask"]["available"] is False
    for key in (
        "ingest",
        "ingest_status",
        "scrape",
        "scrape_urls",
        "scrape_status",
        "pipeline_refresh",
        "jobs",
    ):
        assert links[key]["available"] is True
    assert links["ask"]["auth_required"] is False
    for key in ("search", "stats", "cache_stats", "ingest", "jobs", "docs"):
        assert links[key]["auth_required"] is True
    assert links["cache_clear"]["method"] == "DELETE"
    assert links["cache_expire"]["method"] == "POST"


def test_login_issues_a_bearer_token(client: TestClient) -> None:
    response = client.post(TOKEN_URL, data={"username": USERNAME, "password": PASSWORD})

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 30 * 60
    assert {"self", "api_root", "search"} <= set(body["links"])


def test_login_failures_are_indistinguishable(client: TestClient) -> None:
    wrong_password = client.post(TOKEN_URL, data={"username": USERNAME, "password": "nope"})
    unknown_user = client.post(TOKEN_URL, data={"username": "nobody", "password": PASSWORD})

    assert wrong_password.status_code == 401
    assert unknown_user.status_code == 401
    assert wrong_password.json() == unknown_user.json()
    assert wrong_password.headers["www-authenticate"] == "Bearer"
    assert wrong_password.json()["links"]["token"]["method"] == "POST"


def test_protected_routes_require_a_token(client: TestClient) -> None:
    response = client.get("/api/v1/stats")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert set(response.json()["links"]) == {"api_root", "token"}


def test_a_valid_token_opens_protected_routes(client: TestClient) -> None:
    token = _login(client)

    for path in ("/api/v1/stats", "/api/v1/cache"):
        assert client.get(path, headers=_auth(token)).status_code == 200


def test_invalid_tokens_are_rejected(client: TestClient) -> None:
    expired = _mint(exp=int((datetime.now(UTC) - timedelta(minutes=1)).timestamp()))
    tokens = [_mint() + "x", expired, _mint(typ="refresh")]

    for token in tokens:
        assert client.get("/api/v1/stats", headers=_auth(token)).status_code == 401


def test_docs_are_guarded(client: TestClient) -> None:
    token = _login(client)

    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 401
        assert client.get(path, headers=_auth(token)).status_code == 200


def test_openapi_schema_hides_the_docs_routes(client: TestClient) -> None:
    token = _login(client)
    schema = client.get("/openapi.json", headers=_auth(token)).json()

    assert "/docs" not in schema["paths"]
    assert "/openapi.json" not in schema["paths"]
    assert "/api/v1/search" in schema["paths"]


def test_open_docs_mode(open_client: TestClient) -> None:
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert open_client.get(path).status_code == 200

    links = open_client.get("/api/v1").json()["links"]
    assert links["docs"]["auth_required"] is False


@pytest.mark.parametrize(("method", "path"), PROTECTED_ROUTES)
def test_protected_routes_need_a_token(client: TestClient, method: str, path: str) -> None:
    response = client.request(method, path)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_ask_is_public_and_still_a_stub(client: TestClient) -> None:
    response = client.post("/api/v1/ask")

    assert response.status_code == 501
    assert response.json()["code"] == "not_implemented"
    assert response.json()["links"]["search"]["available"] is True


def test_search_returns_hypermedia_without_embeddings(
    client: TestClient, harness: _Harness
) -> None:
    token = _login(client)

    response = client.post(
        "/api/v1/search",
        json={"query": "how is PAYE withheld", "top_k": 2},
        headers=_auth(token),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query_type"] in {"factual", "calculation", "procedural"}
    assert body["results"]
    assert "embedding" not in body["results"][0]
    assert body["results"][0]["links"]["source"]["href"] == "https://www.ird.gov.lk/paye"
    assert set(body["links"]) == {"self", "stats", "cache_stats", "api_root"}

    event = harness.analytics.events[0]
    assert event["source"] is SearchSource.api
    assert event["search_mode"] is SearchMode.hybrid


def test_invalid_search_payload_returns_the_envelope(client: TestClient) -> None:
    token = _login(client)

    response = client.post(
        "/api/v1/search", json={"query": "x", "top_k": 999}, headers=_auth(token)
    )

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_error"
    assert body["errors"]
    assert body["links"]["api_root"]["href"].endswith("/api/v1")


def test_service_errors_map_to_statuses() -> None:
    harness = _harness(retriever_error=NotFoundError("no documents match"))
    client = TestClient(create_app(api_deps=harness.deps, auth_config=AUTH))
    token = _login(client)

    response = client.post("/api/v1/search", json={"query": "paye"}, headers=_auth(token))

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


def test_unknown_route_uses_the_envelope(client: TestClient) -> None:
    response = client.get("/nope")

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"
    assert "api_root" in response.json()["links"]


def test_auth_config_reports_every_missing_value() -> None:
    settings = Settings(_env_file=None, api_auth_username=USERNAME)

    with pytest.raises(RuntimeError, match="API_AUTH_PASSWORD_HASH"):
        AuthConfig.from_settings(settings)


def test_auth_config_rejects_a_short_secret() -> None:
    settings = Settings(
        _env_file=None,
        api_auth_username=USERNAME,
        api_auth_password_hash=SecretStr(AUTH.password_hash),
        api_auth_secret=SecretStr("too-short"),
    )

    with pytest.raises(RuntimeError, match="at least 32 bytes"):
        AuthConfig.from_settings(settings)


def test_cache_stats_advertise_their_actions(client: TestClient) -> None:
    token = _login(client)

    body = client.get("/api/v1/cache", headers=_auth(token)).json()

    assert body["total_entries"] == 7
    assert body["links"]["cache_clear"]["method"] == "DELETE"
    assert body["links"]["cache_expire"]["method"] == "POST"


def test_cache_mutations_report_counts(client: TestClient) -> None:
    token = _login(client)

    cleared = client.delete("/api/v1/cache", headers=_auth(token))
    expired = client.post("/api/v1/cache/expire", headers=_auth(token))

    assert cleared.status_code == 200
    assert cleared.json()["deleted"] == 7
    assert expired.json()["deleted"] == 2
    assert "cache_stats" in cleared.json()["links"]


def test_enqueue_job_returns_location(client: TestClient) -> None:
    token = _login(client)

    response = client.post(
        JOBS_URL,
        json={"type": "pipeline.refresh", "params": {"full": True}},
        headers=_auth(token),
    )

    assert response.status_code == 202
    assert response.headers["location"].endswith(f"{JOBS_URL}/run-1")
    assert response.json()["run_id"] == "run-1"
    assert response.json()["reused"] is False


def test_list_jobs_returns_an_empty_page(client: TestClient) -> None:
    token = _login(client)

    response = client.get(JOBS_URL, headers=_auth(token))

    assert response.status_code == 200
    assert response.json()["items"] == []
    assert response.json()["links"]["self"]["href"].endswith(JOBS_URL)


def test_job_detail_maps_not_found(client: TestClient) -> None:
    token = _login(client)

    response = client.get(f"{JOBS_URL}/missing", headers=_auth(token))

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


def test_job_cancel_and_replay_return_actions(client: TestClient) -> None:
    token = _login(client)

    cancelled = client.post(f"{JOBS_URL}/run-1/cancel", headers=_auth(token))
    replayed = client.post(f"{JOBS_URL}/run-1/replay", headers=_auth(token))

    assert cancelled.status_code == 202
    assert cancelled.json()["action"] == "cancel"
    assert replayed.status_code == 202
    assert replayed.json()["action"] == "replay"


def test_scrape_and_pipeline_endpoints_enqueue_jobs(client: TestClient) -> None:
    token = _login(client)

    scraped = client.post(
        SCRAPE_URL,
        json={"full": True, "idempotency_key": "scrape-1"},
        headers=_auth(token),
    )
    refreshed = client.post(
        PIPELINE_REFRESH_URL,
        json={"target": "pending", "max_tokens": 1000},
        headers=_auth(token),
    )

    assert scraped.status_code == 202
    assert scraped.headers["location"].endswith(f"{JOBS_URL}/run-1")
    assert refreshed.status_code == 202


def test_ingest_endpoints_enqueue_pending_or_document(client: TestClient) -> None:
    token = _login(client)
    document_id = uuid4()

    pending = client.post(INGEST_URL, json={}, headers=_auth(token))
    document = client.post(
        INGEST_URL,
        json={"document_id": str(document_id)},
        headers=_auth(token),
    )

    assert pending.status_code == 202
    assert document.status_code == 202


def test_scrape_and_ingest_status_endpoints(client: TestClient) -> None:
    token = _login(client)

    scrape = client.get(f"{SCRAPE_URL}/status", headers=_auth(token))
    ingest = client.get(f"{INGEST_URL}/status", headers=_auth(token))
    urls = client.get(f"{SCRAPE_URL}/urls", headers=_auth(token))

    assert scrape.status_code == 200
    assert scrape.json()["total"] == 2
    assert ingest.status_code == 200
    assert ingest.json()["total_documents"] == 3
    assert urls.status_code == 200
    assert urls.json()["items"] == []
