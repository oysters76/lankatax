from __future__ import annotations

from enum import Enum

import pytest
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import ENUM as PgEnum
from sqlalchemy.ext.asyncio import AsyncSession

from lankatax_ai.backend.db import models
from lankatax_ai.backend.db.models import SearchLog
from lankatax_ai.backend.types import (
    ChunkStrategy,
    DocumentKind,
    LlmProvider,
    QueryType,
    SearchMode,
    SearchSource,
)

PERSISTED_ENUMS: list[tuple[str, PgEnum, type[Enum]]] = [
    ("document_kind", models.document_kind_enum, DocumentKind),
    ("chunk_strategy", models.chunk_strategy_enum, ChunkStrategy),
    ("search_source", models.search_source_enum, SearchSource),
    ("search_mode", models.search_mode_enum, SearchMode),
    ("query_type", models.query_type_enum, QueryType),
    ("llm_provider", models.llm_provider_enum, LlmProvider),
]


@pytest.mark.unit
@pytest.mark.parametrize(("name", "pg_enum", "python_enum"), PERSISTED_ENUMS)
def test_db_enum_matches_python_enum(name: str, pg_enum: PgEnum, python_enum: type[Enum]) -> None:
    assert list(pg_enum.enums) == [member.value for member in python_enum], name


@pytest.mark.db
async def test_search_log_accepts_every_llm_provider(
    clean_session: AsyncSession,
) -> None:
    for provider in LlmProvider:
        clean_session.add(
            SearchLog(
                query_text=f"query-{provider.value}",
                query_type=QueryType.factual.value,
                search_mode=SearchMode.hybrid.value,
                result_count=0,
                latency_total_ms=0,
                llm_provider=provider.value,
            )
        )
    await clean_session.commit()

    count = await clean_session.scalar(select(func.count()).select_from(SearchLog))
    assert count == len(LlmProvider)
