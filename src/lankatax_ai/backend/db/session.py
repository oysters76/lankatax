from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession 
from sqlalchemy.orm import sessionmaker 
from lankatax_ai.backend.config import settings 

engine = create_async_engine(settings.connection_string)  
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False) 

async def dispose_engine():
    await engine.dispose() 