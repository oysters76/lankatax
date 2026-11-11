from lankatax_ai.backend.scraper.content_hasher import compute_hash
from lankatax_ai.backend.scraper.crawler import AsyncCrawlerFetchResult
from lankatax_ai.backend.scraper.pdf_extractor import pdf_to_text
from lankatax_ai.backend.scraper.storage import DocumentRecord


def to_document_record(result: AsyncCrawlerFetchResult) -> DocumentRecord | None:
    if result.is_error:
        return None

    if result.parsed_page is not None:
        return DocumentRecord(
            url=result.url,
            title=result.parsed_page.title,
            raw_text=result.parsed_page.text,
            category=result.entry.category.value,
            content_hash=compute_hash(result.parsed_page.text),
        )

    if result.pdf_bytes is not None:
        return DocumentRecord(
            url=result.url,
            title=result.entry.label,
            raw_text=pdf_to_text(result.pdf_bytes),
            category=result.entry.category.value,
            content_hash=compute_hash(result.pdf_bytes),
        )

    return None
