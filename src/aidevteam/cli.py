"""Typer CLI entry point for the aidevteam multi-agent builder."""

import typer
from rich.console import Console
from rich.panel import Panel

from aidevteam.config import get_settings
from aidevteam.graph import build_graph

app = typer.Typer(
    add_completion=False,
    help="aidevteam — Planner → Coder → Tester → Reviewer agents that build Python CLI apps.",
)
console = Console()


@app.command()
def run(
    task: str = typer.Argument(
        ..., help="Natural-language description of the CLI app to build."
    ),
) -> None:
    """Build a Python CLI app from TASK using the agent pipeline.
    """
    settings = get_settings()

    console.print(
        Panel(
            f"[bold]{task}[/bold]",
            title="aidevteam",
            subtitle=f"endpoint={settings.host} · model={settings.resolved_model}",
        )
    )

    graph = build_graph()
    final = graph.invoke({"task": task, "iteration": 0}, {"recursion_limit": 40})

    review = final.get("review")
    report = final.get("test_report")
    workspace = final.get("workspace", "")
    code = final.get("code")

    if review and review.approved:
        status = "[bold green]✔ APPROVED[/bold green]"
    elif review:
        status = "[bold yellow]⚠ max iterations reached — review feedback outstanding[/bold yellow]"
    else:
        status = "[bold red]✘ pipeline ended early[/bold red]"

    lines = [status, ""]
    if workspace:
        lines.append(f"[bold]Workspace:[/bold] {workspace}")
    if code:
        lines.append("[bold]Files:[/bold]")
        lines.extend(f"  • {f.path}" for f in code.files)
    if report:
        lines.append(f"[bold]Tests:[/bold] {report.summary}")
    if review and not review.approved and review.feedback:
        lines.append("[bold]Outstanding feedback:[/bold]")
        lines.extend(f"  - {item}" for item in review.feedback)

    console.print(Panel("\n".join(lines), title="result", border_style="cyan"))


@app.command()
def graph() -> None:
    """Print the agent pipeline as a Mermaid diagram."""
    console.print(build_graph().get_graph().draw_mermaid())


if __name__ == "__main__":
    app()
