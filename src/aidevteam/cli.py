"""Typer CLI entry point for the aidevteam multi-agent builder."""

import typer
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from aidevteam.checkpoint import list_threads, persistent_checkpointer
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


def _approval_loop(graph, config: dict, state: dict) -> dict:
    """Drive the plan-approval gate until the run leaves it.

    The gate interrupts after every (re)generated plan until approved; empty
    rejection feedback aborts the run. Shared by `run` and `resume` so both
    behave identically.
    """
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
    return state


def _render_result(state: dict, run_id: str) -> None:
    """Print the final pipeline result panel."""
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


def _plan_app_name(plan) -> str:
    """The app name of a plan, whether it is a model instance or a plain dict."""
    if isinstance(plan, dict):
        return str(plan.get("app_name") or "")
    return str(getattr(plan, "app_name", "") or "")


def _run_status(snapshot) -> str:
    """Human-readable status of a stored run."""
    if snapshot.interrupts:
        return "[yellow]awaiting approval[/yellow]"
    if snapshot.next:
        return "[cyan]paused[/cyan]"
    return "[green]finished[/green]"


def _truncate(text: str | None, width: int) -> str:
    """One-line, width-limited text for table cells."""
    flat = (text or "").replace("\n", " ").strip()
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def _format_timestamp(timestamp: str) -> str:
    """ISO timestamp -> 'YYYY-MM-DD HH:MM:SS' for display."""
    return timestamp.replace("T", " ")[:19] or "-"


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
    # Resuming an interrupt needs the same thread (thread_id=run_id) + checkpointer.
    config = {"configurable": {"thread_id": run_id}, "recursion_limit": 60}
    with persistent_checkpointer() as saver:
        graph = build_graph(plan_approval=approval, checkpointer=saver)
        state: dict = graph.invoke(
            {
                "task": task,
                "run_id": run_id,
                "iteration": 0,
                # Recorded in the checkpoint so `resume` can rebuild the graph
                # with the same topology the run was started with.
                "plan_approval": approval,
            },
            config,
        )
        state = _approval_loop(graph, config, state)

    _render_result(state, run_id)


@app.command()
def resume(
    run_id: str = typer.Argument(..., help="Run id of the paused run to resume."),
) -> None:
    """Resume a paused run from the persistent checkpoint store.

    The run id is the thread id shown by `list` (and in the `run` header).
    """
    settings = get_settings()
    config = {"configurable": {"thread_id": run_id}, "recursion_limit": 60}

    with persistent_checkpointer() as saver:
        graph = build_graph(plan_approval=settings.plan_approval, checkpointer=saver)
        snapshot = graph.get_state(config)
        values = snapshot.values or {}
        if not values:
            console.print(f"[bold red]✘ unknown run: {run_id}[/bold red]")
            raise typer.Exit(code=1)

        # The gate mode is part of the graph topology: rebuild the graph the
        # way the run was started, otherwise the pending interrupt would be
        # replayed by a node that no longer calls interrupt().
        approval = bool(values.get("plan_approval", settings.plan_approval))
        if approval != settings.plan_approval:
            graph = build_graph(plan_approval=approval, checkpointer=saver)

        console.print(
            Panel(
                f"[bold]{values.get('task', '')}[/bold]",
                title="aidevteam resume",
                subtitle=f"run={run_id} · paused at {', '.join(snapshot.next) or 'end'}",
            )
        )

        if snapshot.interrupts:
            # A pending approval: the interrupt payload carries the plan.
            state = _approval_loop(
                graph, config, {"__interrupt__": list(snapshot.interrupts)}
            )
        elif snapshot.next:
            # Interrupted mid-pipeline (e.g. the process died): LangGraph
            # re-runs the unfinished node, whose side effects are not
            # idempotent — hence the warning.
            console.print(
                "[bold yellow]run paused mid-pipeline — the unfinished node runs again[/bold yellow]"
            )
            state = _approval_loop(graph, config, graph.invoke(None, config))
        else:
            console.print(f"[bold]run {run_id} has already finished[/bold]")
            _render_result(values, run_id)
            return

    _render_result(state, run_id)


@app.command("list")
def list_runs(
    limit: int = typer.Option(20, "--limit", help="Maximum number of runs to list."),
) -> None:
    """List the runs stored in the persistent checkpoint store."""
    with persistent_checkpointer() as saver:
        threads = list_threads(saver)
        if not threads:
            console.print("[bold]no runs recorded yet[/bold]")
            return

        # get_state() only reads the checkpoint, so the gate mode is irrelevant.
        graph = build_graph(plan_approval=False, checkpointer=saver)
        table = Table(title="runs")
        table.add_column("run id", style="bold")
        table.add_column("app")
        table.add_column("task")
        table.add_column("status")
        table.add_column("updated")

        for thread_id, updated_at in threads[:limit]:
            snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
            values = snapshot.values or {}
            table.add_row(
                thread_id,
                _plan_app_name(values.get("plan")) or "-",
                _truncate(values.get("task"), 40),
                _run_status(snapshot),
                _format_timestamp(updated_at),
            )
        console.print(table)


@app.command()
def graph() -> None:
    """Print the agent pipeline as a Mermaid diagram."""
    console.print(
        build_graph(checkpointer=MemorySaver()).get_graph().draw_mermaid()
    )


if __name__ == "__main__":
    app()
