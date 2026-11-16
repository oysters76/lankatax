import asyncio
import time

import typer
from langchain_core.language_models import BaseChatModel
from langchain_openai import OpenAIEmbeddings
from rich.console import Console
from rich.table import Table

from lankatax_ai.backend.config import settings
from lankatax_ai.backend.db.session import async_session, dispose_engine
from lankatax_ai.backend.rag.context_builder import ContextBuilder
from lankatax_ai.backend.rag.query_preprocessor import QueryPreprocessor
from lankatax_ai.backend.rag.reranker import Reranker
from lankatax_ai.backend.rag.retriever import HybridRetriever
from lankatax_ai.backend.rag.search_analytics import SearchAnalytics, SearchTiming
from lankatax_ai.backend.rag.types import SearchMode
from lankatax_ai.backend.types import LlmProvider
from lankatax_ai.cli.state import cli_state

console = Console()


def _build_llm() -> BaseChatModel:
    provider = cli_state.llm_provider
    if provider is LlmProvider.gemini:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.google_api_key.get_secret_value(),
        )
    elif provider is LlmProvider.groq:
        from langchain_groq import ChatGroq

        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=settings.groq_api_key.get_secret_value(),
        )
    else:
        console.print(f"[red]Unknown LLM provider: {provider}. Use gemini or groq.[/red]")
        raise typer.Exit(1)


def search(
    query: str = typer.Argument(None, help="Search query text"),
    mode: str = typer.Option("hybrid", "--mode", "-m", help="Search mode: hybrid, vector, keyword"),
    category: str | None = typer.Option(None, "--category", "-c", help="Filter by tax category"),
    rerank: bool = typer.Option(False, "--rerank", "-r", help="Enable LLM reranking"),
    context: bool = typer.Option(False, "--context", help="Show assembled context window"),
    stats: bool = typer.Option(False, "--stats", help="Show search analytics"),
):
    """Search the tax knowledge base."""
    if stats:
        _show_stats()
        return

    if not query:
        console.print("[red]Query is required unless using --stats.[/red]")
        raise typer.Exit(1)

    try:
        search_mode = SearchMode(mode)
    except ValueError:
        console.print(f"[red]Invalid mode: {mode}. Use hybrid, vector, or keyword.[/red]")
        raise typer.Exit(1)

    asyncio.run(_search_async(query, search_mode, category, rerank, context))


def _show_stats():
    asyncio.run(_show_stats_async())


async def _show_stats_async():
    async with async_session() as session:
        analytics = SearchAnalytics(session)
        stats = await analytics.get_stats()
    await dispose_engine()

    if stats.total_queries == 0:
        console.print("[yellow]No search logs yet.[/yellow]")
        return

    table = Table(title="Search Analytics", show_lines=False, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total queries", f"[cyan]{stats.total_queries}[/cyan]")
    table.add_row("Avg latency", f"[dim]{stats.avg_latency_ms:.0f}ms[/dim]")
    table.add_row("Avg top score", f"[dim]{stats.avg_top_score:.3f}[/dim]")
    hit_color = "green" if stats.hit_rate >= 0.8 else "yellow" if stats.hit_rate >= 0.5 else "red"
    table.add_row("Hit rate", f"[{hit_color}]{stats.hit_rate:.1%}[/{hit_color}]")

    console.print(table)

    if stats.no_result_queries:
        fail_table = Table(title="Top Failed Queries", show_lines=False, padding=(0, 2))
        fail_table.add_column("#", style="dim", width=3, justify="right")
        fail_table.add_column("Query", style="red")

        for i, q in enumerate(stats.no_result_queries, 1):
            fail_table.add_row(str(i), q)

        console.print(fail_table)


async def _search_async(
    query: str, mode: SearchMode, category: str | None, do_rerank: bool, show_context: bool
):
    top_k = cli_state.top_k
    search_timings = SearchTiming()
    search_start_timing, search_end_timing, rerank_end_timing = time.perf_counter(), None, None
    query_preprocessor = QueryPreprocessor()
    processed_query = query_preprocessor.process(query)
    embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
    async with async_session() as session:
        retriever = HybridRetriever(session, embed_model, vector_weight=cli_state.vector_weight)
        results = await retriever.search(
            processed_query.expanded,
            mode=mode,
            top_k=top_k,
            category=category,
            include_embeddings=show_context,
        )
        search_end_timing = time.perf_counter()

    if not results:
        search_timings.record(search_start=search_start_timing, search_end=search_end_timing)
        async with async_session() as log_session:
            analytics = SearchAnalytics(log_session)
            await analytics.log_search(
                query_text=query,
                expanded_query=processed_query.expanded,
                query_type=processed_query.query_type,
                search_mode=mode,
                top_score=None,
                result_count=0,
                timing=search_timings,
            )
            await log_session.commit()
        console.print("[yellow]No results found.[/yellow]")
        await dispose_engine()
        return

    if do_rerank:
        llm = _build_llm()
        reranker = Reranker(llm, threshold=cli_state.rerank_threshold)
        results = await reranker.rerank(query, results, top_k=top_k)
        rerank_end_timing = time.perf_counter()

    search_timings.record(
        search_start=search_start_timing, search_end=search_end_timing, rerank_end=rerank_end_timing
    )
    _print_results_table(results, search_timings)

    if show_context:
        builder = ContextBuilder(
            token_budget=cli_state.token_budget,
            dedup_threshold=cli_state.dedup_threshold,
        )
        context_window = builder.build(results)
        console.print("\n[bold]Assembled Context:[/bold]")
        console.print(context_window)

    async with async_session() as log_session:
        analytics = SearchAnalytics(log_session)
        await analytics.log_search(
            query_text=query,
            expanded_query=processed_query.expanded,
            query_type=processed_query.query_type,
            search_mode=mode,
            top_score=results[0].score if results else None,
            result_count=len(results),
            timing=search_timings,
            llm_provider=cli_state.llm_provider if do_rerank else None,
        )
        await log_session.commit()
    await dispose_engine()


def _print_results_table(results, timing: SearchTiming):
    table = Table(title="Search Results", show_lines=True, padding=(0, 1))
    table.add_column("Rank", style="bold", width=4, justify="right")
    table.add_column("Score", width=6, justify="right")
    table.add_column("Preview", max_width=60)
    table.add_column("Source", style="cyan", max_width=30)
    table.add_column("Mode", style="magenta", width=8)

    for i, r in enumerate(results, 1):
        preview = r.content[:120].replace("\n", " ") + ("..." if len(r.content) > 120 else "")
        source = r.metadata.get("document_title", "Unknown")
        mode_label = _score_label(r)

        table.add_row(
            str(i),
            f"{r.score:.3f}",
            preview,
            source,
            mode_label,
        )

    console.print(table)
    parts = [f"{len(results)} results in {timing.elapsed_seconds:.2f}s"]
    parts.append(f"search={timing.search_ms}ms")
    if timing.rerank_ms:
        parts.append(f"rerank={timing.rerank_ms}ms")
    console.print(f"\n[dim]{' · '.join(parts)}[/dim]")


def _score_label(r) -> str:
    if r.vector_score is not None and r.keyword_score is not None:
        return "hybrid"
    elif r.vector_score is not None:
        return "vector"
    else:
        return "keyword"
