import asyncio
from uuid import UUID

import typer
from langchain_openai import OpenAIEmbeddings
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from lankatax_ai.backend.config import settings
from lankatax_ai.backend.db.session import async_session, dispose_engine
from lankatax_ai.backend.ingestion.pipeline import (
    IngestionResult,
    get_ingestion_status,
    ingest_all,
    ingest_document,
    ingest_pending,
)
from lankatax_ai.backend.ingestion.types import ChunkConfig
from lankatax_ai.backend.scraper.storage import get_document_by_id


def ingest(
    all_docs: bool = typer.Option(False, "--all-docs", help="Re-ingest all documents"),
    document_id: str = typer.Option(None, "--document-id", help="Ingest a specific document by ID"),
    status: bool = typer.Option(False, "--status", help="Show ingestion status"),
):
    """Chunk, embed and store documents into pgvector."""
    if status:
        _show_status()
    else:
        _do_ingest(all_docs=all_docs, document_id=document_id)


def _show_status():
    asyncio.run(_show_status_async())


async def _show_status_async():
    console = Console()
    async with async_session() as session:
        s = await get_ingestion_status(session)

    if s.total_documents == 0:
        console.print("[dim]No documents in database.[/dim]")
    else:
        table = Table(title="Ingestion Status", show_lines=False, padding=(0, 2))
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("Total documents", f"[cyan]{s.total_documents}[/cyan]")
        table.add_row("Ingested", f"[green]{s.ingested}[/green]")
        table.add_row("Pending", f"[yellow]{s.pending}[/yellow]")

        console.print(table)
    await dispose_engine()


def _do_ingest(all_docs: bool, document_id: str | None):
    try:
        asyncio.run(_do_ingest_async(all_docs, document_id))
    except KeyboardInterrupt:
        Console().print("[bold yellow]Ingestion interrupted.[/bold yellow]")


async def _do_ingest_async(all_docs: bool, document_id: str | None):
    console = Console()
    model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=settings.openai_api_key.get_secret_value(),
    )
    config = ChunkConfig()

    async with async_session() as session:
        if document_id:
            result = await _ingest_single(session, UUID(document_id), config, model, console)
        elif all_docs:
            result = await _ingest_batch(session, config, model, console, mode="all")
        else:
            result = await _ingest_batch(session, config, model, console, mode="pending")

    _print_summary(console, result)
    await dispose_engine()


async def _ingest_single(session, doc_id, config, model, console):
    doc = await get_document_by_id(session, doc_id)
    if doc is None:
        console.print(f"[red]Document {doc_id} not found.[/red]")
        return IngestionResult(errors=[f"Document {doc_id} not found"])

    console.print(f"Ingesting [cyan]{doc.title}[/cyan]...")
    chunk_count, tokens_used = await ingest_document(session, doc, config, model)
    return IngestionResult(docs_processed=1, chunks_created=chunk_count, tokens_used=tokens_used)


async def _ingest_batch(session, config, model, console, mode):
    fn = ingest_all if mode == "all" else ingest_pending
    label = "all documents" if mode == "all" else "pending documents"

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed} docs"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(f"Ingesting {label}", total=None)

        def on_progress(doc_id: UUID, chunk_count: int) -> None:
            progress.advance(task)

        result = await fn(session, config, model, on_progress=on_progress)

    return result


def _print_summary(console, result):
    console.print()
    table = Table(title="Ingestion Summary", show_lines=False, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Documents processed", f"[green]{result.docs_processed}[/green]")
    table.add_row("Chunks created", f"[cyan]{result.chunks_created}[/cyan]")
    table.add_row("Tokens used", f"[dim]{result.tokens_used:,}[/dim]")
    cost = result.tokens_used * settings.embedding_cost_per_token
    table.add_row("Estimated cost", f"[yellow]${cost:.6f}[/yellow]")

    if result.errors:
        table.add_row("Errors", f"[red]{len(result.errors)}[/red]")
        for err in result.errors:
            table.add_row("", f"[dim]{err}[/dim]")

    console.print(table)
