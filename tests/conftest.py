import os

os.environ["TESTCONTAINERS_RYUK_DISABLED"] = "true"

import pytest
import pytest_asyncio
from pathlib import Path
from sqlalchemy import text, create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from testcontainers.community.postgres import PostgresContainer

SCHEMA_PATH = Path(__file__).parent.parent / "src/lankatax_ai/backend/db/init_schema.sql"
MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations"


@pytest.fixture(scope="session")
def pg_container():
    with PostgresContainer(image="pgvector/pgvector:pg17") as pg:
        yield pg


@pytest.fixture(scope="session")
def db_url(pg_container):
    return pg_container.get_connection_url()


@pytest.fixture(scope="session")
def async_engine(db_url):
    # Synchronously seed the database schema
    sync_engine = create_engine(db_url)
    with sync_engine.begin() as conn:
        conn.execute(text(SCHEMA_PATH.read_text()))
        for migration in sorted(MIGRATIONS_DIR.glob("*.sql")):
            sql = migration.read_text().strip()
            lines = [l.strip() for l in sql.splitlines() if l.strip() and not l.strip().startswith("--")]
            if not lines:
                continue 
            conn.execute(text(sql))
    sync_engine.dispose()

    # Create and yield the async engine
    engine = create_async_engine(db_url.replace("psycopg2", "asyncpg"), echo=True)
    yield engine

    # Properly close async engine connection pools on test suite exit
    # (Since this is a synchronous fixture, we don't await engine.dispose())
    engine.sync_engine.dispose()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def session(async_engine):
    async with AsyncSession(async_engine) as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture(loop_scope="session")
async def clean_session(async_engine):
    """Per-test session that rolls back after each test — no data leaks."""
    async with AsyncSession(async_engine) as session:
        await session.execute(text("DELETE FROM documents"))
        await session.commit()
        yield session
