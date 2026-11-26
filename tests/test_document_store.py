from sqlalchemy import select

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
