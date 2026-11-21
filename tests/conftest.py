import os

os.environ["TESTCONTAINERS_RYUK_DISABLED"] = "true"

# Placeholder settings so importing lankatax_ai.backend.config never needs a
# real .env. Tests that exercise Settings pass _env_file=None.
_SETTINGS_DEFAULTS = {
    "POSTGRES_DB_NAME": "lankatax_test",
    "POSTGRES_USER": "lankatax",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "OPENAI_API_KEY": "test",
    "GOOGLE_API_KEY": "test",
    "GROQ_API_KEY": "test",
    "LANGSMITH_API_KEY": "test",
    "LANGSMITH_PROJECT": "lankatax-test",
    "LANGSMITH_TRACING": "false",
    "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com",
}
for _name, _value in _SETTINGS_DEFAULTS.items():
    os.environ.setdefault(_name, _value)

from pathlib import Path

import pytest
import pytest_asyncio
from alembic.config import Config as AlembicConfig
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from testcontainers.community.postgres import PostgresContainer

from alembic import command as alembic_command
from lankatax_ai.backend.db.models import Base

ALEMBIC_INI = Path(__file__).parent.parent / "alembic.ini"
_TABLES = [t.name for t in Base.metadata.sorted_tables]


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


async def _truncate(session: AsyncSession) -> None:
    await session.execute(text(f"TRUNCATE TABLE {', '.join(_TABLES)} CASCADE"))


@pytest_asyncio.fixture(loop_scope="session")
async def clean_session(async_engine):
    """Per-test session; empties every table before the test runs."""
    async with AsyncSession(async_engine) as session:
        await _truncate(session)
        await session.commit()
        yield session
