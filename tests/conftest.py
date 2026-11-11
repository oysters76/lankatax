import pytest
import pytest_asyncio
from pathlib import Path
from sqlalchemy import text, create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from testcontainers.community.postgres import PostgresContainer

SCHEMA_PATH = Path(__file__).parent.parent / "src/lankatax_ai/backend/db/init_schema.sql"


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
    sync_engine.dispose()

    # Create and yield the async engine
    engine = create_async_engine(db_url.replace("psycopg2", "asyncpg"), echo=True)
    yield engine
    
    # Properly close async engine connection pools on test suite exit
    # (Since this is a synchronous fixture, we don't await engine.dispose())
    engine.sync_engine.dispose()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def session(async_engine):
    # Both scope (pytest life) and loop_scope (asyncio life) must be set to "session"
    async with AsyncSession(async_engine) as session:
        yield session
        await session.rollback()
