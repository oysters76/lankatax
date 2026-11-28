from urllib.parse import unquote, urlparse, urlunparse

from lankatax_ai.backend.scraper.registry import ScrapableURL
from lankatax_ai.backend.scraper.types import AsyncCrawlerFetchResult
from lankatax_ai.backend.types import DocumentKind


def get_pdf_name_from_link(url: str) -> str:
    path = urlparse(url).path
    filename = unquote(path.split("/")[-1])
    return filename.removesuffix(".pdf") or filename


def _normalize_url(url: str, base_host: str) -> str | None:
    """Return an https URL on the allowed host, or ``None`` for anything else."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return None

    allowed = urlparse(base_host if "://" in base_host else f"//{base_host}").hostname
    if not allowed:
        return None
    allowed = allowed.lower().removeprefix("www.")

    host = (parsed.hostname or "").lower()
    if not host or not (host == allowed or host.endswith(f".{allowed}")):
        return None

    # The registry is https-only; upgrading keeps one canonical URL per file.
    return urlunparse(("https", parsed.netloc, parsed.path, parsed.params, parsed.query, ""))


def discover_pdf_links(
    fetch_results: list[AsyncCrawlerFetchResult],
    known_urls: set[str],
    registry_urls: set[str],
    base_host: str = "ird.gov.lk",
) -> list[ScrapableURL]:
    pdf_links = []
    seen = set()
    for fr in fetch_results:
        if fr.parsed_page is None:
            continue
        for link in fr.parsed_page.pdf_links:
            normalized = _normalize_url(link, base_host)
            if normalized is None:
                continue
            if normalized in seen:
                continue
            if normalized in known_urls:
                continue
            if normalized in registry_urls:
                continue
            pdf_links.append(
                ScrapableURL(
                    url=normalized,
                    category=fr.entry.category,
                    label=get_pdf_name_from_link(normalized),
                    kind=DocumentKind.pdf,
                )
            )
            seen.add(normalized)
    return pdf_links
