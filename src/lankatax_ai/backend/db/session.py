from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from lankatax_ai.backend.config import settings

engine = create_async_engine(
    settings.connection_string,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    pool_recycle=3600,
)
async_session = async_sessionmaker(engine, expire_on_commit=False)


async def dispose_engine() -> None:
    await engine.dispose()
