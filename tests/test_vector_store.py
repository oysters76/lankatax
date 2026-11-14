from sqlalchemy import func, select

from lankatax_ai.backend.db.models import Chunk as ChunkModel
from lankatax_ai.backend.db.models import Document
from lankatax_ai.backend.ingestion.vector_store import (
    delete_chunks_by_document_id,
    delete_chunks_by_url,
    search_similar,
    store_chunks,
)
from lankatax_ai.backend.scraper.storage import save_documents
from tests.helpers import make_chunk, make_document


async def _insert_doc(session, **overrides):
    doc = make_document(**overrides)
    await save_documents(session, [doc])
    row = (await session.execute(select(Document.id).where(Document.url == doc.url))).scalar_one()
    return row


async def test_store_and_count_chunks(clean_session):
    doc_id = await _insert_doc(clean_session)
    chunks = [make_chunk(content=f"chunk {i}") for i in range(3)]
    await store_chunks(clean_session, doc_id, chunks)
    await clean_session.commit()

    cnt = (
        await clean_session.execute(
            select(func.count()).select_from(ChunkModel).where(ChunkModel.document_id == doc_id)
        )
    ).scalar_one()
    assert cnt == 3


async def test_store_chunks_empty_list(clean_session):
    doc_id = await _insert_doc(clean_session)
    await store_chunks(clean_session, doc_id, [])
    await clean_session.commit()

    cnt = (
        await clean_session.execute(
            select(func.count()).select_from(ChunkModel).where(ChunkModel.document_id == doc_id)
        )
    ).scalar_one()
    assert cnt == 0


async def test_delete_chunks_by_document_id(clean_session):
    doc_id = await _insert_doc(clean_session)
    chunks = [make_chunk(content=f"chunk {i}") for i in range(3)]
    await store_chunks(clean_session, doc_id, chunks)
    await clean_session.commit()

    deleted = await delete_chunks_by_document_id(clean_session, doc_id)
    await clean_session.commit()
    assert deleted == 3

    cnt = (
        await clean_session.execute(
            select(func.count()).select_from(ChunkModel).where(ChunkModel.document_id == doc_id)
        )
    ).scalar_one()
    assert cnt == 0


async def test_delete_chunks_by_url(clean_session):
    url = "https://www.ird.gov.lk/delete-test"
    doc_id = await _insert_doc(clean_session, url=url)
    chunks = [make_chunk(content=f"chunk {i}") for i in range(2)]
    await store_chunks(clean_session, doc_id, chunks)
    await clean_session.commit()

    deleted = await delete_chunks_by_url(clean_session, url)
    await clean_session.commit()
    assert deleted == 2


async def test_search_similar(clean_session):
    doc_id = await _insert_doc(clean_session)

    near = [1.0] + [0.0] * 1535
    far = [0.0] * 1535 + [1.0]

    chunks = [
        make_chunk(content="near chunk", embedding=near),
        make_chunk(content="far chunk", embedding=far),
    ]
    await store_chunks(clean_session, doc_id, chunks)
    await clean_session.commit()

    query_embedding = [1.0] + [0.0] * 1535
    results = await search_similar(clean_session, query_embedding, top_k=2)

    assert len(results) == 2
    assert results[0]["content"] == "near chunk"
    assert results[1]["content"] == "far chunk"
    assert results[0]["distance"] < results[1]["distance"]
