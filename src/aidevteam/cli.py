"""Typer CLI entry point for the aidevteam multi-agent builder."""

import typer
from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel

from aidevteam.config import get_settings
from aidevteam.graph import build_graph
from aidevteam.workspace import new_run_id, workspace_for

app = typer.Typer(
    add_completion=False,
    help="aidevteam — Planner → Coder → Tester → Reviewer agents that build Python CLI apps.",
)
console = Console()


def _show_plan(plan: dict, revision: int = 0) -> None:
    """Render a generated plan for the manual approval prompt."""
    lines = [f"[bold]App:[/bold] {plan.get('app_name', '?')}", ""]
    summary = plan.get("summary")
    if summary:
        lines.append(summary)
        lines.append("")
    commands = plan.get("cli_commands") or []
    if commands:
        lines.append("[bold]Example commands:[/bold]")
        lines.extend(f"  • {command}" for command in commands)
    lines.append("[bold]Files:[/bold]")
    lines.extend(
        f"  • {spec.get('path')} — {spec.get('description')}"
        for spec in plan.get("files") or []
    )
    console.print(
        Panel(
            "\n".join(lines),
            title="plan approval",
            subtitle=f"revision {revision}" if revision else "first draft",
            border_style="magenta",
        )
    )


@app.command()
def run(
    task: str = typer.Argument(
        ..., help="Natural-language description of the CLI app to build."
    ),
    plan_approval: bool | None = typer.Option(
        None,
        "--plan-approval/--no-plan-approval",
        help="Require human approval of the plan before coding. Default: follow the plan_approval setting.",
    ),
) -> None:
    """Build a Python CLI app from TASK using the agent pipeline."""
    settings = get_settings()
    run_id = new_run_id()

    console.print(
        Panel(
            f"[bold]{task}[/bold]",
            title="aidevteam",
            subtitle=f"run={run_id} · endpoint={settings.host} · model={settings.resolved_model}",
        )
    )

    # plan_approval=True means a human must approve the plan before coding;
    # an explicitly passed flag overrides the settings.
    approval = settings.plan_approval if plan_approval is None else plan_approval
    graph = build_graph(plan_approval=approval)
    # Resuming an interrupt needs the same thread (thread_id=run_id) + checkpointer.
    config = {"configurable": {"thread_id": run_id}, "recursion_limit": 60}
    state: dict = graph.invoke(
        {"task": task, "run_id": run_id, "iteration": 0}, config
    )

    # The gate interrupts after every (re)generated plan until approved;
    # empty rejection feedback aborts the run.
    while state.get("__interrupt__"):
        payload = state["__interrupt__"][0].value or {}
        _show_plan(payload.get("plan") or {}, payload.get("revision", 0))
        if typer.confirm("Approve this plan?"):
            resume = {"approved": True, "feedback": ""}
        else:
            feedback = typer.prompt(
                "What should the Planner change? (empty input aborts)", default=""
            ).strip()
            if not feedback:
                console.print(
                    "[bold red]✘ run aborted — plan rejected without feedback[/bold red]"
                )
                raise typer.Exit(code=1)
            resume = {"approved": False, "feedback": feedback}
        state = graph.invoke(Command(resume=resume), config)

    review = state.get("review")
    report = state.get("test_report")
    workspace = str(workspace_for(run_id))
    code = state.get("code")

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
