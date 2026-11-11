import typer
from rich import print as rprint

from lankatax_ai.cli.scrape import scrape

app = typer.Typer(name="lankatax", help="LankaTax AI - Sri Lanka Tax RAG Pipeline")
app.command()(scrape)


@app.command()
def ingest(
    all_docs: bool = typer.Option(False, help="Ingest all documents"),
    pending: bool = typer.Option(True, help="Ingest only pending documents"),
    document_id: str = typer.Option(None, help="Ingest a specific document by ID"),
):
    """
    Ingests documents into the pipeline or a specific document
    """
    rprint(
        f"[bold green]✓[/] ingest documents "
        f"(all={all_docs} pending={pending}, document_id={document_id})"
    )


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
