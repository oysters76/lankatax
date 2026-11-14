import os

os.environ["TESTCONTAINERS_RYUK_DISABLED"] = "true"

from pathlib import Path

import pytest
import pytest_asyncio
from alembic.config import Config as AlembicConfig
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from testcontainers.community.postgres import PostgresContainer

from alembic import command as alembic_command

ALEMBIC_INI = Path(__file__).parent.parent / "alembic.ini"


@pytest.fixture(scope="session")
def pg_container():
    with PostgresContainer(image="pgvector/pgvector:pg17") as pg:
        yield pg


@pytest.fixture(scope="session")
def db_url(pg_container):
    return pg_container.get_connection_url()


@pytest.fixture(scope="session")
def async_engine(db_url):
    alembic_cfg = AlembicConfig(str(ALEMBIC_INI))
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    alembic_command.upgrade(alembic_cfg, "head")

    engine = create_async_engine(db_url.replace("psycopg2", "asyncpg"), echo=True)

    yield engine

    engine.sync_engine.dispose()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def session(async_engine):
    async with AsyncSession(async_engine) as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture(loop_scope="session")
async def clean_session(async_engine):
    """Per-test session that rolls back after each test — no data leaks."""
    from lankatax_ai.backend.db.models import Document

    async with AsyncSession(async_engine) as session:
        await session.execute(delete(Document))
        await session.commit()
        yield session
