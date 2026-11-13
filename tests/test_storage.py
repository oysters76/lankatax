from lankatax_ai.backend.scraper.storage import (
    get_hash_map,
    get_scrape_status,
    save_documents,
)
from tests.helpers import make_document_record as _make_record


async def test_save_and_retrieve(clean_session):
    """Save one document, then verify it exists via get_hash_map."""
    drec = _make_record()
    await save_documents(clean_session, [drec])
    hashmap = await get_hash_map(clean_session)
    assert drec.url in hashmap
    assert drec.content_hash == hashmap[drec.url].content_hash


async def test_upsert_updates_existing(clean_session):
    """Save same URL twice with different content — should update, not duplicate."""
    drec1 = _make_record()
    drec2 = _make_record(raw_text="some new tax content", content_hash="bcd213")

    await save_documents(clean_session, [drec1])
    await save_documents(clean_session, [drec2])

    hashmap = await get_hash_map(clean_session)

    assert drec1.url in hashmap
    assert drec2.url in hashmap
    assert drec1.content_hash != hashmap[drec1.url].content_hash
    assert drec2.content_hash == hashmap[drec2.url].content_hash


async def test_get_hash_map_multiple(clean_session):
    """Save multiple records, verify hash map has all of them."""
    drec1 = _make_record(url="https://www.ird.gov.lk/test1", content_hash="a")
    drec2 = _make_record(url="https://www.ird.gov.lk/test2", content_hash="b")
    drec3 = _make_record(url="https://www.ird.gov.lk/test3", content_hash="c")
    drec4 = _make_record(url="https://www.ird.gov.lk/test4", content_hash="d")
    drecs = [drec1, drec2, drec3, drec4]

    await save_documents(clean_session, drecs)
    hashmap = await get_hash_map(clean_session)
    assert len(hashmap.keys()) == len(drecs)
    for dr in drecs:
        assert dr.content_hash == hashmap[dr.url].content_hash


async def test_get_scrape_status_empty(clean_session):
    """Status on empty table — total should be 0."""
    result = await get_scrape_status(clean_session)
    assert result.total == 0


async def test_get_scrape_status_after_save(clean_session):
    """Save records, then verify status counts."""
    drec1 = _make_record(url="https://www.ird.gov.lk/test1", content_hash="a")
    drec2 = _make_record(url="https://www.ird.gov.lk/test2", content_hash="b")
    await save_documents(clean_session, [drec1, drec2])
    result = await get_scrape_status(clean_session)
    assert result.total == 2
    assert result.recent == 2
    assert result.first_scrape is not None
    assert result.last_scrape is not None
