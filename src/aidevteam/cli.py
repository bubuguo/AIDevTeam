"""Typer CLI entry point for the aidevteam multi-agent builder."""

import typer
from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from aidevteam.checkpoint import list_threads, persistent_checkpointer
from aidevteam.config import get_settings
from aidevteam.github_issue import GitHubIssueError, get_github_issue_tool
from aidevteam.github_workflow import format_task_body
from aidevteam.graph import build_graph
from aidevteam.workspace import workspace_for

app = typer.Typer(
    add_completion=False,
    help="aidevteam — Planner → Coder → Tester → Reviewer agents that build Python CLI apps.",
)
console = Console()


def _create_github_issue(task: str) -> int:
    """Create a GitHub issue and return its number as the canonical issue ID.

    A GitHub issue is required because its number is the only identifier used
    for the workflow thread, workspace, and node jobs. The issue body carries
    the task in the versioned workflow format, so a later ``resume`` can
    rebuild the run's context from the issue alone.
    """
    settings = get_settings()
    if not settings.github_token or not settings.github_owner or not settings.github_repo:
        raise GitHubIssueError(
            "GitHub issue tracking is required; set GITHUB_TOKEN, GITHUB_OWNER, "
            "and GITHUB_REPO"
        )

    title = f"{task[:60]}..." if len(task) > 60 else task
    body = format_task_body(task)
    with get_github_issue_tool() as tool:
        issue = tool.create_issue(title, body)
        return issue.number


def _render_result(state: dict, issue_id: int) -> None:
    """Print the final pipeline result panel."""
    review = state.get("review")
    report = state.get("test_report")
    workspace = str(workspace_for(issue_id))
    code = state.get("code")

    if review and review.approved:
        status = "[bold green]✔ APPROVED[/bold green]"
    elif review:
        status = "[bold yellow]⚠ max iterations reached — review feedback outstanding[/bold yellow]"
    else:
        status = "[bold red]✘ pipeline ended early[/bold red]"

    lines = [status, ""]
    lines.append(f"[bold]Issue:[/bold] #{issue_id}")
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


def _run_status(snapshot) -> str:
    """Human-readable status of a stored run."""
    if snapshot.next and "approve_plan" in snapshot.next:
        return "[yellow]awaiting approval[/yellow]"
    if snapshot.next:
        return "[cyan]paused[/cyan]"
    return "[green]finished[/green]"


def _format_timestamp(timestamp: str) -> str:
    """ISO timestamp -> 'YYYY-MM-DD HH:MM:SS' for display."""
    return timestamp.replace("T", " ")[:19] or "-"


@app.command()
def run(
    task: str = typer.Argument(
        ..., help="Natural-language description of the CLI app to build."
    ),
) -> None:
    """Build a Python CLI app from TASK using the agent pipeline.

    The task becomes a GitHub issue; the plan is posted as a comment on that
    issue and the run waits there for a human to comment ``/approve`` or
    ``/reject <reason>``.
    """
    try:
        issue_id = _create_github_issue(task)
    except GitHubIssueError as exc:
        console.print(f"[bold red]✘ GitHub operation failed: {exc}[/bold red]")
        raise typer.Exit(code=1) from exc
    
    settings = get_settings()
    console.print(
        Panel(
            f"[bold]{task}[/bold]",
            title="aidevteam",
            subtitle=f"issue={issue_id} · endpoint={settings.host} · model={settings.resolved_model}",
        )
    )

    # Every task plan must be approved by a human, on the GitHub issue.
    # Resuming needs the same thread (thread_id=issue_id) + checkpointer.
    config = {"configurable": {"thread_id": str(issue_id)}, "recursion_limit": 60}
    with persistent_checkpointer() as saver:
        graph = build_graph(checkpointer=saver)
        payload: dict = {
            "issue_id": issue_id,
            "iteration": 0,
        }
        state: dict = graph.invoke(payload, config)

    _render_result(state, issue_id)


@app.command()
def resume(
    issue_id: int = typer.Argument(..., help="GitHub issue ID of the paused run to resume."),
) -> None:
    """Resume a paused run from the persistent checkpoint store.

    The GitHub issue ID is the thread ID shown by `list` (and in the `run` header).
    The run picks up where it stopped: if it was waiting for a plan decision,
    it waits for the next ``/approve`` or ``/reject`` comment on the issue; if
    it died inside a node, that node runs again.
    """
    config = {"configurable": {"thread_id": str(issue_id)}, "recursion_limit": 60}

    with persistent_checkpointer() as saver:
        # Every task plan must be approved by a human.
        graph = build_graph(checkpointer=saver)
        snapshot = graph.get_state(config)
        values = snapshot.values or {}
        if not values:
            console.print(f"[bold red]✘ unknown issue: {issue_id}[/bold red]")
            raise typer.Exit(code=1)

        stored_issue_id = values.get("issue_id")
        if stored_issue_id != issue_id:
            console.print(
                f"[bold red]✘ checkpoint for issue {issue_id} is incompatible: "
                "missing or mismatched issue_id[/bold red]"
            )
            raise typer.Exit(code=1)

        console.print(
            Panel(
                f"[bold]issue #{issue_id}[/bold]",
                title="aidevteam resume",
                subtitle=f"paused at {', '.join(snapshot.next) or 'end'}",
            )
        )

        if not snapshot.next:
            console.print(f"[bold]issue {issue_id} has already finished[/bold]")
            _render_result(values, issue_id)
            return

        if "approve_plan" not in snapshot.next:
            # Interrupted mid-pipeline (e.g. the process died): LangGraph
            # re-runs the unfinished node, whose side effects are not
            # idempotent — hence the warning.
            console.print(
                "[bold yellow]run paused mid-pipeline — the unfinished node runs again[/bold yellow]"
            )

        state = graph.invoke(None, config)

    _render_result(state, issue_id)


@app.command("list")
def list_runs(
    limit: int = typer.Option(20, "--limit", help="Maximum number of issues to list."),
) -> None:
    """List the issues stored in the persistent checkpoint store."""
    with persistent_checkpointer() as saver:
        threads = list_threads(saver)
        if not threads:
            console.print("[bold]no issues recorded yet[/bold]")
            return

        # get_state() only reads the checkpoint, so the gate mode is irrelevant.
        graph = build_graph(checkpointer=saver)
        table = Table(title="issues")
        table.add_column("issue id", style="bold")
        table.add_column("status")
        table.add_column("updated")

        for thread_id, updated_at in threads[:limit]:
            snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
            values = snapshot.values or {}
            stored_issue_id = values.get("issue_id")
            if str(stored_issue_id) != thread_id:
                continue
            table.add_row(
                thread_id,
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
