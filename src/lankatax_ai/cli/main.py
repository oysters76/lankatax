import logging

import typer
from rich import print as rprint

from lankatax_ai.cli.cache import cache
from lankatax_ai.cli.ingest import ingest
from lankatax_ai.cli.scrape import scrape
from lankatax_ai.cli.search import search
from lankatax_ai.cli.state import cli_state

app = typer.Typer(name="lankatax", help="LankaTax AI - Sri Lanka Tax RAG Pipeline")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    llm: str = typer.Option(
        "gemini", "--llm", help="LLM provider: gemini, groq, deepseek or openai"
    ),
    token_budget: int | None = typer.Option(
        None, "--token-budget", help="Context token budget (auto from model if unset)"
    ),
    rerank_threshold: int = typer.Option(3, "--rerank-threshold", help="Min rerank score 1-10"),
    vector_weight: float = typer.Option(
        0.7, "--vector-weight", help="Hybrid vector weight 0.0-1.0"
    ),
    dedup_threshold: float = typer.Option(
        0.6, "--dedup-threshold", help="Dedup cosine threshold 0.0-1.0"
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Default results count"),
):
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    cli_state.llm_provider = llm
    if token_budget is not None:
        cli_state.token_budget = token_budget
    cli_state.rerank_threshold = rerank_threshold
    cli_state.vector_weight = vector_weight
    cli_state.dedup_threshold = dedup_threshold
    cli_state.top_k = top_k


app.command()(scrape)
app.command()(ingest)
app.command()(search)
app.command()(cache)


@app.command()
def chat():
    """
    Opens an interactive chat repl
    """
    rprint("[bold green]✓[/] starting interactive chat REPL")


@app.command("eval")
def run_eval(
    run: bool = typer.Option(False, help="Run evaluation suite"),
    report: bool = typer.Option(False, help="Generate evaluation report"),
):
    """
    Runs the evaluation suite
    """
    rprint(f"[bold green]✓[/] evaluation suite (run={run}, report={report})")


@app.command()
def graph():
    """
    Exports the multi agent graph as a mermaid diagram
    """
    rprint("[bold green]✓[/] exporting LangGraph as Mermaid")
