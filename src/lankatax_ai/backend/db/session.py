from pgvector.asyncpg import register_vector
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from lankatax_ai.backend.config import settings

engine = create_async_engine(settings.connection_string)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@event.listens_for(engine.sync_engine, "connect")
def _register_vector(dbapi_conn, connection_record):
    dbapi_conn.run_async(register_vector)


async def dispose_engine():
    await engine.dispose()
