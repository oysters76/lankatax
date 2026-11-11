import asyncio

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from lankatax_ai.backend.db.session import async_session
from lankatax_ai.backend.scraper.crawler import AsyncCrawler, AsyncCrawlerFetchResult
from lankatax_ai.backend.scraper.registry import URLRegistry
from lankatax_ai.backend.scraper.storage import get_hash_map, save_documents
from lankatax_ai.backend.scraper.transform import to_document_record


def scrape(
    full: bool = typer.Option(False, "--full", help="Force re-ingest all"),
    list_urls: bool = typer.Option(False, "--list", help="Show registry URLs"),
    status: bool = typer.Option(False, "--status", help="Show scrape status"),
):
    """Crawl ird.gov.lk — incremental by default, --full for all, --status for last run."""
    if list_urls:
        _list_urls()
    elif status:
        _show_status()
    else:
        _do_scrape(full)


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
    pass


def _do_scrape(full: bool = False):
    registry = URLRegistry()
    metrics = {"succeeded": 0, "failed": 0}
    results = []
    interrupted = False
    total = len(registry.get_all())

    try:
        asyncio.run(_do_scrape_async(registry, metrics, results))
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
        saved = asyncio.run(_save_results(results, full))
        Console().print(f"[green]Saved {saved} documents to DB[/green]")


async def _save_results(results: list[AsyncCrawlerFetchResult], full: bool) -> int:
    records = [r for r in (to_document_record(res) for res in results) if r is not None]
    if not records:
        return 0
    async with async_session() as session:
        if full:
            to_save = records
        else:
            hash_map = await get_hash_map(session)
            to_save = [
                r
                for r in records
                if r.url not in hash_map or r.content_hash != hash_map[r.url].content_hash
            ]
        if not to_save:
            return 0
        await save_documents(session, to_save)
    return len(to_save)


async def _do_scrape_async(
    registry, metrics: dict[str, int], results: list[AsyncCrawlerFetchResult]
) -> None:
    entries = registry.get_all()
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
