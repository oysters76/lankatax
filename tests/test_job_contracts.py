from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError as PydanticValidationError

from lankatax_ai.backend.jobs import contracts, policies

pytestmark = pytest.mark.unit


def test_ingest_input_defaults():
    payload = contracts.IngestInput()

    assert payload.target == "pending"
    assert payload.max_documents is None
    assert payload.max_tokens is None
    assert payload.batch_size == 25


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_documents": 0},
        {"max_tokens": 0},
        {"batch_size": 0},
        {"batch_size": 101},
        {"target": "document"},
    ],
)
def test_ingest_input_validation(kwargs):
    with pytest.raises(PydanticValidationError):
        contracts.IngestInput(**kwargs)


def test_ingest_document_input_requires_uuid():
    document_id = uuid4()
    payload = contracts.IngestDocumentInput(document_id=document_id)

    assert payload.document_id == document_id


def test_refresh_input_accepts_budget_and_strategy():
    payload = contracts.RefreshInput(
        full=True,
        target="all",
        max_documents=10,
        max_tokens=5000,
        batch_size=5,
    )

    assert payload.full is True
    assert payload.target == "all"
    assert payload.max_tokens == 5000


def test_outputs_round_trip():
    scrape = contracts.ScrapeOutput(
        saved=2,
        skipped=1,
        failed=0,
        discovered=3,
        partial=False,
        duration_ms=120,
    )
    ingest = contracts.IngestOutput(
        docs_processed=2,
        chunks_created=4,
        tokens_used=100,
        partial=True,
        budget_exhausted=True,
        errors=["budget"],
        duration_ms=200,
    )
    cache = contracts.CacheOutput(expired=3, duration_ms=5)

    assert scrape.model_dump()["saved"] == 2
    assert ingest.model_dump()["budget_exhausted"] is True
    assert cache.model_dump()["expired"] == 3


def test_policies_are_derived_from_settings():
    assert policies.SCRAPE_RETRIES == 3
    assert policies.INGEST_RETRIES == 3
    assert policies.CACHE_RETRIES == 2
    assert policies.SCRAPE_EXECUTION_TIMEOUT == timedelta(seconds=1800)
    assert policies.INGEST_EXECUTION_TIMEOUT == timedelta(seconds=7200)
    assert policies.SCHEDULE_TIMEOUT == timedelta(seconds=3600)
