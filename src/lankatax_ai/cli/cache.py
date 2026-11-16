import asyncio

import typer
from rich.console import Console
from rich.table import Table

from lankatax_ai.backend.db.session import async_session, dispose_engine
from lankatax_ai.backend.rag.result_cache import PgResultCache

console = Console()


def cache(
    stats: bool = typer.Option(False, "--stats", help="Show cache statistics"),
    clear: bool = typer.Option(False, "--clear", help="Flush all cached results"),
):
    """Manage the semantic query cache."""
    if stats:
        _show_stats()
    elif clear:
        _clear_cache()
    else:
        console.print("[dim]Use --stats or --clear.[/dim]")


def _show_stats():
    asyncio.run(_show_stats_async())


async def _show_stats_async():
    async with async_session() as session:
        rc = PgResultCache(session)
        stats = await rc.get_stats()
    await dispose_engine()

    if stats.total_entries == 0:
        console.print("[yellow]Cache is empty.[/yellow]")
        return

    table = Table(title="Cache Statistics", show_lines=False, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Cached queries", f"[cyan]{stats.total_entries}[/cyan]")
    table.add_row("Total hits", f"[green]{stats.total_hits}[/green]")
    hit_color = "green" if stats.hit_rate >= 0.5 else "yellow" if stats.hit_rate >= 0.2 else "red"
    table.add_row("Hit rate", f"[{hit_color}]{stats.hit_rate:.1%}[/{hit_color}]")
    table.add_row("Oldest entry", f"[dim]{stats.oldest_entry_hours:.1f}h ago[/dim]")

    console.print(table)


def _clear_cache():
    asyncio.run(_clear_cache_async())


async def _clear_cache_async():
    async with async_session() as session:
        rc = PgResultCache(session)
        count = await rc.clear()
        await session.commit()
    await dispose_engine()

    if count == 0:
        console.print("[dim]Cache was already empty.[/dim]")
    else:
        console.print(f"[green]Cleared {count} cached entries.[/green]")
