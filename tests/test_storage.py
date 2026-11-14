from lankatax_ai.backend.scraper.storage import (
    get_hash_map,
    get_scrape_status,
    save_documents,
)
from tests.helpers import make_document


async def test_save_and_retrieve(clean_session):
    """Save one document, then verify it exists via get_hash_map."""
    doc = make_document()
    await save_documents(clean_session, [doc])
    await clean_session.commit()
    hashmap = await get_hash_map(clean_session)
    assert doc.url in hashmap
    assert doc.content_hash == hashmap[doc.url]


async def test_upsert_updates_existing(clean_session):
    """Save same URL twice with different content — should update, not duplicate."""
    doc1 = make_document()
    doc2 = make_document(raw_text="some new tax content", content_hash="bcd213")

    await save_documents(clean_session, [doc1])
    await clean_session.commit()
    await save_documents(clean_session, [doc2])
    await clean_session.commit()

    hashmap = await get_hash_map(clean_session)

    assert doc1.url in hashmap
    assert doc2.url in hashmap
    assert doc1.content_hash != hashmap[doc1.url]
    assert doc2.content_hash == hashmap[doc2.url]


async def test_get_hash_map_multiple(clean_session):
    """Save multiple records, verify hash map has all of them."""
    doc1 = make_document(url="https://www.ird.gov.lk/test1", content_hash="a")
    doc2 = make_document(url="https://www.ird.gov.lk/test2", content_hash="b")
    doc3 = make_document(url="https://www.ird.gov.lk/test3", content_hash="c")
    doc4 = make_document(url="https://www.ird.gov.lk/test4", content_hash="d")
    docs = [doc1, doc2, doc3, doc4]

    await save_documents(clean_session, docs)
    await clean_session.commit()
    hashmap = await get_hash_map(clean_session)
    assert len(hashmap.keys()) == len(docs)
    for d in docs:
        assert d.content_hash == hashmap[d.url]


async def test_get_scrape_status_empty(clean_session):
    """Status on empty table — total should be 0."""
    result = await get_scrape_status(clean_session)
    assert result.total == 0


async def test_get_scrape_status_after_save(clean_session):
    """Save records, then verify status counts."""
    doc1 = make_document(url="https://www.ird.gov.lk/test1", content_hash="a")
    doc2 = make_document(url="https://www.ird.gov.lk/test2", content_hash="b")
    await save_documents(clean_session, [doc1, doc2])
    await clean_session.commit()
    result = await get_scrape_status(clean_session)
    assert result.total == 2
    assert result.recent == 2
    assert result.first_scrape is not None
    assert result.last_scrape is not None
