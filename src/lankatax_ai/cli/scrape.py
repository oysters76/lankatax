import asyncio

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from lankatax_ai.backend.db.session import async_session, dispose_engine
from lankatax_ai.backend.scraper.crawler import AsyncCrawler, AsyncCrawlerFetchResult
from lankatax_ai.backend.scraper.pdf_discovery import discover_pdf_links
from lankatax_ai.backend.scraper.registry import URLRegistry
from lankatax_ai.backend.scraper.storage import (
    get_hash_map,
    get_known_urls,
    get_scrape_status,
    save_documents,
)
from lankatax_ai.backend.scraper.transform import to_document


def scrape(
    full: bool = typer.Option(False, "--full", help="Force re-ingest all"),
    list_urls: bool = typer.Option(False, "--list", help="Show registry URLs"),
    status: bool = typer.Option(False, "--status", help="Show scrape status"),
    no_discover: bool = typer.Option(
        False, "--no-discover", help="Turn off auto-discovery of pdfs"
    ),
):
    """Crawl ird.gov.lk — incremental by default, --full for all, --status for last run."""
    if list_urls:
        _list_urls()
    elif status:
        _show_status()
    else:
        _do_scrape(full, no_discover)


def _list_urls():
    console = Console()
    registry = URLRegistry()

    table = Table(title="URL Registry")
    table.add_column("URL", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Label", style="yellow")
    table.add_column("Type", style="magenta")

    for entry in registry.get_all():
        table.add_row(
            entry.url,
            entry.category.value,
            entry.label,
            entry.kind.value,
        )

    console.print(table)


def _show_status():
    asyncio.run(_show_status_async())


async def _show_status_async():
    console = Console()
    async with async_session() as session:
        status = await get_scrape_status(session)

    if status.total == 0:
        console.print("[dim]No documents scraped yet.[/dim]")
    else:
        table = Table(title="Scrape Status", show_lines=False, padding=(0, 2))
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("Total documents", f"[cyan]{status.total}[/cyan]")
        table.add_row("Scraped (last 24h)", f"[green]{status.recent}[/green]")
        table.add_row("First scrape", f"[dim]{status.first_scrape:%Y-%m-%d %H:%M}[/dim]")
        table.add_row("Last scrape", f"[dim]{status.last_scrape:%Y-%m-%d %H:%M}[/dim]")

        console.print(table)
    await dispose_engine()


def _do_scrape(full: bool = False, no_discover: bool = False):
    registry = URLRegistry()
    metrics = {"succeeded": 0, "failed": 0}
    results = []
    interrupted = False
    total = len(registry.get_all())

    try:
        asyncio.run(_do_scrape_async(registry.get_all(), metrics, results))
        if not no_discover:
            discovered = asyncio.run(_discover_and_crawl_pdfs(results, registry, metrics))
            results.extend(discovered)
    except KeyboardInterrupt:
        interrupted = True
    _print_scrape_summary_table(metrics, len(results))

    if interrupted and results:
        console = Console()
        msg = f"⚠ Crawl interrupted — {len(results)}/{total} pages completed"
        console.print(f"[bold yellow]{msg}[/bold yellow]")
        save = typer.confirm("Save partial results?")
        if not save:
            return

    if results:
        saved = asyncio.run(_save_and_dispose(results, full))
        Console().print(f"[green]Saved {saved} documents to DB[/green]")


async def _discover_and_crawl_pdfs(
    results: list[AsyncCrawlerFetchResult],
    registry: URLRegistry,
    metrics: dict[str, int],
) -> list[AsyncCrawlerFetchResult]:
    console = Console()

    async with async_session() as session:
        known = await get_known_urls(session)

    registry_urls = {entry.url for entry in registry.get_all()}
    discovered = discover_pdf_links(results, known, registry_urls)

    if not discovered:
        console.print("[dim]No new PDFs discovered.[/dim]")
        await dispose_engine()
        return []

    console.print(f"[cyan]Discovered {len(discovered)} new PDFs — crawling...[/cyan]")

    pdf_results: list[AsyncCrawlerFetchResult] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Crawling discovered PDFs", total=len(discovered))

        def on_result(result: AsyncCrawlerFetchResult) -> None:
            pdf_results.append(result)
            if result.is_error:
                metrics["failed"] += 1
            else:
                metrics["succeeded"] += 1
            progress.advance(task)

        async with AsyncCrawler() as crawler:
            await crawler.crawl_all(discovered, on_result=on_result)

    await dispose_engine()
    return pdf_results


async def _save_and_dispose(results: list[AsyncCrawlerFetchResult], full: bool) -> int:
    saved = await _save_results(results, full)
    await dispose_engine()
    return saved


async def _save_results(results: list[AsyncCrawlerFetchResult], full: bool) -> int:
    records = [r for r in (to_document(res) for res in results) if r is not None]
    if not records:
        return 0
    async with async_session() as session:
        if full:
            to_save = records
        else:
            hash_map = await get_hash_map(session)
            to_save = [
                r for r in records if r.url not in hash_map or r.content_hash != hash_map[r.url]
            ]
        if not to_save:
            return 0
        await save_documents(session, to_save)
    return len(to_save)


async def _do_scrape_async(
    entries, metrics: dict[str, int], results: list[AsyncCrawlerFetchResult]
) -> None:
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Crawling ird.gov.lk", total=len(entries))

        def on_result(result: AsyncCrawlerFetchResult) -> None:
            results.append(result)
            if result.is_error:
                metrics["failed"] += 1
            else:
                metrics["succeeded"] += 1
            progress.advance(task)

        async with AsyncCrawler() as crawler:
            await crawler.crawl_all(entries, on_result=on_result)


def _print_scrape_summary_table(metrics, total):
    console = Console()
    console.print()

    table = Table(title="Scrape Summary", show_lines=False, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Count", justify="right")

    table.add_row("Succeeded", f"[green]{metrics['succeeded']}[/green]")
    failed_str = f"[red]{metrics['failed']}[/red]" if metrics["failed"] else "[dim]0[/dim]"
    table.add_row("Failed", failed_str)
    table.add_row("Total", f"[bold]{total}[/bold]")

    console.print(table)
