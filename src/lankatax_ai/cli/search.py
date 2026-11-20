import asyncio
import time
from datetime import date as date_type

import typer
from langchain_core.language_models import BaseChatModel
from langchain_openai import OpenAIEmbeddings
from rich.console import Console
from rich.table import Table

from lankatax_ai.backend.config import settings
from lankatax_ai.backend.db.session import async_session, dispose_engine
from lankatax_ai.backend.rag.context_builder import ContextBuilder
from lankatax_ai.backend.rag.metadata_filter import MetadataFilters
from lankatax_ai.backend.rag.multi_query_retriever import MultiQueryExpander
from lankatax_ai.backend.rag.query_preprocessor import QueryPreprocessor
from lankatax_ai.backend.rag.reranker import Reranker
from lankatax_ai.backend.rag.result_cache import PgResultCache
from lankatax_ai.backend.rag.retriever import HybridRetriever
from lankatax_ai.backend.rag.search_analytics import SearchAnalytics, SearchTiming
from lankatax_ai.backend.rag.types import SearchMode
from lankatax_ai.backend.types import DocumentKind, LlmProvider
from lankatax_ai.backend.utils import merge_result_sets
from lankatax_ai.cli.state import cli_state

console = Console()


def _parse_date(value: str | None) -> date_type | None:
    if value is None:
        return None
    try:
        return date_type.fromisoformat(value)
    except ValueError:
        raise typer.BadParameter(f"Invalid date: {value!r}. Use YYYY-MM-DD.")


def _parse_doc_type(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        DocumentKind(value)
    except ValueError:
        raise typer.BadParameter(
            f"Invalid doc type: {value!r}. Choose from {[k.value for k in DocumentKind]}."
        )
    return value


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
    doc_type: str | None = typer.Option(
        None,
        "--doc-type",
        "-d",
        callback=_parse_doc_type,
        help="Filter by document type: pdf, page",
    ),
    after: str | None = typer.Option(
        None, "--after", callback=_parse_date, help="Filter docs scraped after date (YYYY-MM-DD)"
    ),
    before: str | None = typer.Option(
        None, "--before", callback=_parse_date, help="Filter docs scraped before date (YYYY-MM-DD)"
    ),
    rerank: bool = typer.Option(False, "--rerank", "-r", help="Enable LLM reranking"),
    context: bool = typer.Option(False, "--context", help="Show assembled context window"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass semantic cache"),
    multi_query: bool = typer.Option(
        False, "--multi-query", help="Expand query via LLM for better recall"
    ),
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

    filters = MetadataFilters(category=category, doc_type=doc_type, after=after, before=before)

    asyncio.run(_search_async(query, search_mode, filters, rerank, context, no_cache, multi_query))


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
    cache_pct = stats.cache_hit_count / stats.total_queries * 100
    table.add_row(
        "Cache hits", f"[green]{stats.cache_hit_count}[/green] [dim]({cache_pct:.0f}%)[/dim]"
    )
    table.add_row("Multi-query", f"[blue]{stats.multi_query_count}[/blue]")

    console.print(table)

    if stats.no_result_queries:
        fail_table = Table(title="Top Failed Queries", show_lines=False, padding=(0, 2))
        fail_table.add_column("#", style="dim", width=3, justify="right")
        fail_table.add_column("Query", style="red")

        for i, q in enumerate(stats.no_result_queries, 1):
            fail_table.add_row(str(i), q)

        console.print(fail_table)


async def _search_async(
    query: str,
    mode: SearchMode,
    filters: MetadataFilters,
    do_rerank: bool,
    show_context: bool,
    no_cache: bool = False,
    multi_query: bool = False,
):
    top_k = cli_state.top_k
    search_timings = SearchTiming()
    search_start_timing, search_end_timing, rerank_end_timing = time.perf_counter(), None, None
    query_preprocessor = QueryPreprocessor()
    processed_query = query_preprocessor.process(query)
    embed_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=settings.openai_api_key.get_secret_value(),
    )
    cache_hit = False

    # Step 1: check cache
    query_embedding = await embed_model.aembed_query(processed_query.expanded)
    if not no_cache:
        async with async_session() as session:
            rc = PgResultCache(session)
            cached = await rc.get(processed_query.expanded, query_embedding)
            if cached:
                cache_hit = True
                results = cached
                search_end_timing = time.perf_counter()
                console.print("[dim]Cache hit[/dim]")

    # Step 2: on miss, run full pipeline
    if not cache_hit:
        if multi_query:
            llm = _build_llm()
            expander = MultiQueryExpander(llm)
            alt_queries = await expander.expand(processed_query.expanded)
            all_queries = [processed_query.expanded, *alt_queries]
            console.print(f"[dim]Multi-query: searching {len(all_queries)} queries[/dim]")
            async with async_session() as session:
                retriever = HybridRetriever(
                    session, embed_model, vector_weight=cli_state.vector_weight
                )
                result_sets = [
                    await retriever.search(
                        q,
                        mode=mode,
                        top_k=top_k,
                        filters=filters,
                        include_embeddings=show_context,
                    )
                    for q in all_queries
                ]
            results = merge_result_sets(result_sets)[:top_k]
            search_end_timing = time.perf_counter()
        else:
            async with async_session() as session:
                retriever = HybridRetriever(
                    session, embed_model, vector_weight=cli_state.vector_weight
                )
                results = await retriever.search(
                    processed_query.expanded,
                    mode=mode,
                    top_k=top_k,
                    filters=filters,
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
                cache_hit=cache_hit,
                multi_query=multi_query,
            )
            await log_session.commit()
        console.print("[yellow]No results found.[/yellow]")
        await dispose_engine()
        return

    if do_rerank and not cache_hit:
        llm = _build_llm()
        reranker = Reranker(llm, threshold=cli_state.rerank_threshold)
        results = await reranker.rerank(query, results, top_k=top_k)
        rerank_end_timing = time.perf_counter()

    search_timings.record(
        search_start=search_start_timing,
        search_end=search_end_timing,
        rerank_end=rerank_end_timing,
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

    # Step 3: cache results on miss
    if not cache_hit and not no_cache:
        async with async_session() as cache_session:
            rc = PgResultCache(cache_session)
            await rc.put(processed_query.expanded, query_embedding, results)
            await cache_session.commit()

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
            cache_hit=cache_hit,
            multi_query=multi_query,
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
