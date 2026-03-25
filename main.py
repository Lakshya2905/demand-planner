"""
Demand Planning & Forecasting System
Entry point with CLI interface.
"""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data import DemandDataPreprocessor
from src.tools.forecasting import set_demand_data
from src.agents.demand_agent import create_demand_agent
from src.agents.report_agent import ReportAgent
from src.utils import get_logger

logger = get_logger(__name__)
console = Console()


def load_data(data_path: str):
    """Load and preprocess demand data."""
    preprocessor = DemandDataPreprocessor()
    raw = preprocessor.load(data_path)
    data = preprocessor.preprocess(raw)
    set_demand_data(data)
    return data


@click.group()
def cli():
    """Demand Planning & Forecasting System powered by Claude and LangChain."""
    pass


@cli.command()
@click.option(
    "--data",
    default="sample_data/demand_data.csv",
    help="Path to demand data CSV file.",
)
def demo(data: str):
    """Run a demo with sample data."""
    data_path = Path(data)

    # Generate sample data if it does not exist
    if not data_path.exists():
        console.print("[yellow]Sample data not found. Generating...[/yellow]")
        from sample_data.generate import generate_sample_data

        generate_sample_data()

    console.print(Panel("Loading demand data...", style="blue"))
    df = load_data(data_path)

    preprocessor = DemandDataPreprocessor()
    summary = preprocessor.summary(df)

    console.print(Panel(f"""
[bold]Dataset Summary[/bold]
Rows: {summary['total_rows']}
SKUs: {summary['sku_count']}
Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}
Mean Demand: {summary['demand_stats']['mean']:.2f}
    """, title="Demo Data", style="green"))

    console.print("\n[bold]Available SKUs:[/bold]")
    for sku in preprocessor.list_skus(df):
        console.print(f"  - {sku}")

    console.print(
        "\n[dim]Run with --interactive to ask questions, "
        "or --report --sku SKU-ID to generate forecasts.[/dim]"
    )


@cli.command()
@click.option(
    "--data",
    default="sample_data/demand_data.csv",
    help="Path to demand data CSV file.",
)
def interactive(data: str):
    """Start interactive natural language query mode."""
    data_path = Path(data)

    if not data_path.exists():
        console.print("[yellow]Sample data not found. Generating...[/yellow]")
        from sample_data.generate import generate_sample_data

        generate_sample_data()

    console.print(Panel("Starting Demand Planning Assistant...", style="blue"))
    load_data(data_path)

    agent = create_demand_agent(verbose=True)
    chat_history = []

    console.print(
        "\n[bold green]Demand Planning Assistant Ready[/bold green]\n"
        "Ask questions about your inventory demand data.\n"
        "Type 'quit' or 'exit' to stop.\n"
    )

    while True:
        try:
            query = console.input("[bold blue]> [/bold blue]")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        if not query.strip():
            continue

        try:
            result = agent.invoke(
                {"input": query, "chat_history": chat_history}
            )
            output = result.get("output", "No response generated.")
            console.print(f"\n[green]{output}[/green]\n")

            # Maintain chat history
            from langchain_core.messages import HumanMessage, AIMessage

            chat_history.append(HumanMessage(content=query))
            chat_history.append(AIMessage(content=output))

        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")
            logger.error(f"Agent error: {e}", exc_info=True)


@cli.command()
@click.option(
    "--data",
    default="sample_data/demand_data.csv",
    help="Path to demand data CSV file.",
)
@click.option("--sku", required=True, help="SKU identifier to generate report for.")
@click.option("--horizon", default=30, help="Forecast horizon in days.")
@click.option("--output", default=None, help="Output file path for the report.")
def report(data: str, sku: str, horizon: int, output: str):
    """Generate a demand forecast report for a specific SKU."""
    data_path = Path(data)

    if not data_path.exists():
        console.print("[yellow]Sample data not found. Generating...[/yellow]")
        from sample_data.generate import generate_sample_data

        generate_sample_data()

    console.print(Panel(f"Generating report for {sku}...", style="blue"))

    report_agent = ReportAgent(data_path)
    report_text = report_agent.generate_single_report(sku, horizon)

    if output:
        output_path = Path(output)
        output_path.write_text(report_text)
        console.print(f"[green]Report saved to {output_path}[/green]")
    else:
        console.print(report_text)


if __name__ == "__main__":
    cli()
