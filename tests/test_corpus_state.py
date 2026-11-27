from __future__ import annotations

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from lankatax_ai.backend.db.models import CacheEntry, CorpusState

pytestmark = pytest.mark.db


async def test_corpus_state_is_seeded_as_a_singleton(clean_session: AsyncSession) -> None:
    state = await clean_session.get(CorpusState, 1)

    assert state is not None
    assert state.epoch == 0

    clean_session.add(CorpusState(id=2, epoch=0))
    with pytest.raises(IntegrityError):
        await clean_session.flush()
    await clean_session.rollback()


async def test_corpus_epoch_can_advance(clean_session: AsyncSession) -> None:
    state = await clean_session.get(CorpusState, 1)
    assert state is not None

    state.epoch += 1
    await clean_session.commit()
    await clean_session.refresh(state)

    assert state.epoch == 1


async def test_cache_entries_default_to_epoch_zero(clean_session: AsyncSession) -> None:
    clean_session.add(CacheEntry(query_text="paye", response="[]"))
    await clean_session.commit()

    entry = (await clean_session.execute(select(CacheEntry))).scalar_one()
    assert entry.corpus_epoch == 0


async def test_scrape_runs_table_is_removed(clean_session: AsyncSession) -> None:
    with pytest.raises(ProgrammingError):
        await clean_session.execute(text("SELECT 1 FROM scrape_runs"))
    await clean_session.rollback()
