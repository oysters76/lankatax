from urllib.parse import unquote, urlparse

from lankatax_ai.backend.scraper.crawler import AsyncCrawlerFetchResult
from lankatax_ai.backend.scraper.registry import DocumentKind, ScrapableURL


def get_pdf_name_from_link(url: str) -> str:
    path = urlparse(url).path
    filename = unquote(path.split("/")[-1])
    return filename.removesuffix(".pdf") or filename


def discover_pdf_links(
    fetch_results: list[AsyncCrawlerFetchResult],
    known_urls: set[str],
    registry_urls: set[str],
) -> list[ScrapableURL]:
    pdf_links = []
    seen = set()
    for fr in fetch_results:
        if fr.parsed_page is None:
            continue
        for link in fr.parsed_page.pdf_links:
            if link in seen:
                continue
            if link in known_urls:
                continue
            if link in registry_urls:
                continue
            pdf_links.append(
                ScrapableURL(
                    url=link,
                    category=fr.entry.category,
                    label=get_pdf_name_from_link(link),
                    kind=DocumentKind.pdf,
                )
            )
            seen.add(link)
    return pdf_links
