import typer
from rich import print as rprint

from lankatax_ai.cli.ingest import ingest
from lankatax_ai.cli.scrape import scrape

app = typer.Typer(name="lankatax", help="LankaTax AI - Sri Lanka Tax RAG Pipeline")
app.command()(scrape)
app.command()(ingest)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, help="Number of results"),
    mode: str = typer.Option("hybrid", help="Search mode"),
):
    """
    Searches a query against the knowledge base, and limits by k top results
    """
    rprint(f"[bold green]✓[/] search knowledge base (query='{query}', top_k={top_k}, mode={mode})")


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
