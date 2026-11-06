"""Typer CLI entry point for the aidevteam multi-agent builder."""

import typer
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, StateSnapshot
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from aidevteam.checkpoint import list_threads, persistent_checkpointer
from aidevteam.config import get_settings
from aidevteam.github_issue import GitHubIssueError, get_github_issue_tool
from aidevteam.github_workflow import parse_task_body
from aidevteam.graph import build_graph
from aidevteam.issuecontext import fetch_issue_context
from aidevteam.workspace import workspace_for

app = typer.Typer(
    add_completion=False,
    help="aidevteam — Planner → Coder → Tester → Reviewer agents that build Python CLI apps.",
)
console = Console()


def _validate_issue_for_run(issue_id: int) -> str:
    """Return the task text of an existing, open GitHub issue.

    Every run starts from an issue a human opened on GitHub; the CLI never
    creates issues. The issue must exist, be open, and carry a non-empty
    task in its body (plain text is enough — the versioned workflow marker
    is optional, see github_workflow.parse_task_body).
    """
    with get_github_issue_tool() as tool:
        try:
            issue = tool.get_issue(issue_id)
        except GitHubIssueError as exc:
            raise GitHubIssueError(f"issue {issue_id} is not readable: {exc}") from exc
    if issue.state != "open":
        raise GitHubIssueError(f"issue {issue_id} is {issue.state}, not open")
    task = parse_task_body(issue.body or "")
    if not task:
        raise GitHubIssueError(
            f"issue {issue_id} has no task text in its body — describe the "
            "CLI app to build in the issue body"
        )
    return task


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


def _pending_interrupt(snapshot: StateSnapshot) -> dict | None:
    """The payload of a pending approval interrupt, if the run waits at the gate.

    The gate interrupts with ``{"issue_id": ..., "plan_comment_id": ...}``
    (see graph.approve_plan); any other pending state is not an approval
    wait.
    """
    for task in snapshot.tasks or ():
        for intr in getattr(task, "interrupts", None) or ():
            value = getattr(intr, "value", None)
            if isinstance(value, dict) and "plan_comment_id" in value:
                return value
    return None


def _decision_from_issue(issue_id: int, payload: dict) -> dict | None:
    """Read the human's decision for a pending approval from the issue.

    Only a decision posted *after* the plan comment recorded in the
    interrupt payload counts, so a verdict on an older draft is never
    replayed. Returns ``None`` while no decision has been posted yet.
    """
    context = fetch_issue_context(int(issue_id))
    decision = context.decision_after(payload["plan_comment_id"])
    if decision is None:
        return None
    if decision.get("action") == "approve":
        return {"action": "approve"}
    return {"action": "reject", "feedback": str(decision.get("feedback") or "")}


def _run_status(snapshot: StateSnapshot) -> str:
    """Human-readable status of a stored run."""
    if _pending_interrupt(snapshot):
        return "[yellow]awaiting approval[/yellow]"
    if snapshot.next:
        return "[cyan]paused[/cyan]"
    return "[green]finished[/green]"


def _format_timestamp(timestamp: str) -> str:
    """ISO timestamp -> 'YYYY-MM-DD HH:MM:SS' for display."""
    return timestamp.replace("T", " ")[:19] or "-"


@app.command()
def run(
    issue_id: int = typer.Argument(
        ..., help="GitHub issue number (opened by a human) whose body describes the app to build."
    ),
) -> None:
    """Build a Python CLI app from an existing GitHub issue.

    All new requests come from GitHub issues: open an issue whose body
    describes the CLI app, then start the pipeline with this command. The
    plan is posted as a comment on that issue and the run pauses there
    (LangGraph interrupt), so this command returns as soon as the plan is
    up. Comment ``/approve`` or ``/reject <reason>`` on the issue, then run
    ``aidevteam resume <issue>`` to continue.
    """
    try:
        task = _validate_issue_for_run(issue_id)
    except GitHubIssueError as exc:
        console.print(f"[bold red]✘ cannot start issue {issue_id}: {exc}[/bold red]")
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
        # Refuse a second pipeline on the same issue: the checkpoint thread
        # already exists, so this run was started before — resume it instead.
        existing = graph.get_state(config)
        if existing.values:
            console.print(
                f"[bold red]✘ issue {issue_id} already has a run "
                f"({_run_status(existing).strip()}) — use "
                f"`aidevteam resume {issue_id}` instead[/bold red]"
            )
            raise typer.Exit(code=1)
        payload: dict = {
            "issue_id": issue_id,
            "iteration": 0,
        }
        try:
            state: dict = graph.invoke(payload, config)
        except (GitHubIssueError, RuntimeError) as exc:
            # e.g. a node job failed, or the issue is unreadable.
            console.print(f"[bold red]✘ pipeline failed: {exc}[/bold red]")
            raise typer.Exit(code=1) from exc

    if state.get("__interrupt__"):
        # The approval gate paused the run: the plan is posted and the
        # checkpoint holds the pending interrupt. Nothing is polling —
        # the human's decision (and a resume) continues the run.
        console.print(
            f"[bold magenta]⏸ plan posted on issue #{issue_id} — "
            "waiting for approval[/bold magenta]\n"
            "Comment `/approve` or `/reject <reason>` on the issue, then run:\n"
            f"  aidevteam resume {issue_id}"
        )
        return

    _render_result(state, issue_id)


@app.command()
def resume(
    issue_id: int = typer.Argument(..., help="GitHub issue ID of the paused run to resume."),
) -> None:
    """Resume a paused run from the persistent checkpoint store.

    The GitHub issue ID is the thread ID shown by `list` (and in the `run` header).
    The run picks up where it stopped: if it is waiting for a plan decision,
    the newest ``/approve`` or ``/reject`` comment on the issue (posted after
    the latest plan) is supplied to the graph; if it died inside a node, that
    node runs again.
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

        pending = _pending_interrupt(snapshot)
        if pending is not None:
            # The run waits at the approval gate: read the human's decision
            # from the issue and hand it to the graph. Nothing is polled —
            # with no decision yet there is simply nothing to resume.
            try:
                decision = _decision_from_issue(issue_id, pending)
            except GitHubIssueError as exc:
                console.print(f"[bold red]✘ resume failed: {exc}[/bold red]")
                raise typer.Exit(code=1) from exc
            if decision is None:
                console.print(
                    f"[bold yellow]no `/approve` or `/reject` comment after the "
                    f"latest plan on issue {issue_id} yet — nothing to resume[/bold yellow]"
                )
                return
            # The verdict itself is announced by the gate node when it
            # records the decision — no duplicate message here.
            try:
                state = graph.invoke(Command(resume=decision), config)
            except (GitHubIssueError, RuntimeError) as exc:
                console.print(f"[bold red]✘ resume failed: {exc}[/bold red]")
                raise typer.Exit(code=1) from exc
        else:
            # Interrupted mid-pipeline (e.g. the process died): LangGraph
            # re-runs the unfinished node, whose side effects are not
            # idempotent — hence the warning.
            console.print(
                "[bold yellow]run paused mid-pipeline — the unfinished node runs again[/bold yellow]"
            )
            try:
                state = graph.invoke(None, config)
            except (GitHubIssueError, RuntimeError) as exc:
                console.print(f"[bold red]✘ resume failed: {exc}[/bold red]")
                raise typer.Exit(code=1) from exc

    if state.get("__interrupt__"):
        # A rejection loops back to the planner, which posts a revised plan
        # and pauses at the gate again — that's a healthy pause, not an error.
        console.print(
            f"[bold magenta]⏸ revised plan posted on issue #{issue_id} — "
            "waiting for approval[/bold magenta]\n"
            "Comment `/approve` or `/reject <reason>` on the issue, then run:\n"
            f"  aidevteam resume {issue_id}"
        )
        return

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
