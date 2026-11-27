from datetime import UTC, datetime

from sqlalchemy import select, update

from lankatax_ai.backend.db.models import Document
from lankatax_ai.backend.documents import DocumentStore
from tests.helpers import make_document


async def _id_for(session, url: str):
    return (await session.execute(select(Document.id).where(Document.url == url))).scalar_one()


async def test_save_and_commit_persists_documents(clean_session):
    store = DocumentStore(clean_session)
    doc = make_document(url="https://www.ird.gov.lk/store1")

    await store.save([doc])
    await store.commit()

    assert await store.known_urls() == {doc.url}


async def test_get_returns_saved_document(clean_session):
    store = DocumentStore(clean_session)
    doc = make_document(url="https://www.ird.gov.lk/store2")
    await store.save([doc])
    await store.commit()

    found = await store.get(await _id_for(clean_session, doc.url))

    assert found is not None
    assert found.url == doc.url
    assert found.content_hash == doc.content_hash


async def test_hash_map_maps_url_to_content_hash(clean_session):
    store = DocumentStore(clean_session)
    doc = make_document(url="https://www.ird.gov.lk/store3", content_hash="hash3")
    await store.save([doc])
    await store.commit()

    assert await store.hash_map() == {doc.url: "hash3"}


async def test_hash_map_can_be_filtered_to_requested_urls(clean_session):
    store = DocumentStore(clean_session)
    a = make_document(url="https://www.ird.gov.lk/hash-a", content_hash="hash-a")
    b = make_document(url="https://www.ird.gov.lk/hash-b", content_hash="hash-b")
    await store.save([a, b])
    await store.commit()

    assert await store.hash_map([a.url]) == {a.url: "hash-a"}
    assert await store.hash_map([]) == {}


async def test_scrape_status_counts_saved_documents(clean_session):
    store = DocumentStore(clean_session)
    await store.save(
        [
            make_document(url="https://www.ird.gov.lk/store4"),
            make_document(url="https://www.ird.gov.lk/store5"),
        ]
    )
    await store.commit()

    status = await store.scrape_status()

    assert status.total == 2
    assert status.recent == 2


async def test_mark_ingested_moves_document_out_of_pending(clean_session):
    store = DocumentStore(clean_session)
    doc = make_document(url="https://www.ird.gov.lk/store6", content_hash="hash6")
    await store.save([doc])
    await store.commit()
    doc_id = await _id_for(clean_session, doc.url)

    assert await store.pending_ids() == [doc_id]

    await store.mark_ingested(doc_id, "hash6")
    await store.commit()

    assert await store.pending_ids() == []
    assert await store.all_ids() == [doc_id]


async def test_rescrape_refreshes_scraped_at(clean_session):
    store = DocumentStore(clean_session)
    url = "https://www.ird.gov.lk/store7"
    doc = make_document(url=url, content_hash="old-hash")
    await store.save([doc])
    await store.commit()
    doc_id = await _id_for(clean_session, url)

    old = datetime(2020, 1, 1, tzinfo=UTC)
    await clean_session.execute(
        update(Document).where(Document.id == doc_id).values(scraped_at=old, updated_at=old)
    )
    await clean_session.commit()

    await store.save([make_document(url=url, content_hash="new-hash")])
    await store.commit()

    found = await store.get(doc_id)
    assert found is not None
    assert found.content_hash == "new-hash"
    assert found.scraped_at > old
    assert found.updated_at > old
