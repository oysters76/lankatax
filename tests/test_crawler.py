from sqlalchemy import text


async def test_db_connected(session):
    result = await session.execute(text("SELECT 1")) 
    assert result.scalar() == 1 