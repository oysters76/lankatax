import dataclasses

from langchain_openai import OpenAIEmbeddings

from lankatax_ai.backend.ingestion.types import Chunk


async def embed_chunks(model: OpenAIEmbeddings, chunks: list[Chunk]) -> list[Chunk]:
    texts: list[str] = [c.content for c in chunks]
    vecs: list[list[float]] = await model.aembed_documents(texts)
    results: list[Chunk] = []
    for chunk, vec in zip(chunks, vecs):
        c_cloned = dataclasses.replace(chunk, embedding=vec)
        results.append(c_cloned)
    return results
