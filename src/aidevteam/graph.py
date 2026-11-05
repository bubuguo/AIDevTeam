"""LangGraph wiring for the Planner → Coder → Tester → Reviewer pipeline.

START → planner → approve_plan → coder → tester → reviewer ─→ approved → END
                ↑                    ↑                 │
                └── plan rejected ───┘                 └── revisions ──┘

The run is driven by its GitHub issue: the task lives in the issue body,
every plan draft is posted as an issue comment, and the human approves or
rejects a plan by commenting ``/approve`` or ``/reject <reason>``.

The task, the plan and the feedback are *transient* context. They are read
from GitHub whenever a node needs them and are never written to the
checkpoint, so a checkpoint stays small and a run can be resumed from the
issue alone. The human's verdict is the exception: it is persisted as
``plan_approved`` so the conditional edge after the approval gate can route
on it.
"""

import time

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from rich.console import Console

from aidevteam.agents.planner import planner
from aidevteam.config import get_settings
from aidevteam.issuecontext import fetch_issue_context
from aidevteam.rpc import call_node
from aidevteam.state import TeamState

console = Console()


def _route_after_review(state: TeamState) -> str:
    """Send approved work to END; otherwise loop back to the coder."""
    settings = get_settings()
    if state.get("approved"):
        return END
    if state.get("iteration", 0) >= settings.max_iterations:
        return END
    return "coder"


def _route_after_plan_approval(state: TeamState) -> str:
    """Send an approved plan to the coder; otherwise back to the planner.

    The gate records the human's decision in ``plan_approved``, so the
    branch is driven by persisted state rather than by an in-memory
    routing instruction. A missing flag counts as "not approved".
    """
    return "coder" if state.get("plan_approved") else "planner"


def approve_plan(state: TeamState) -> dict:
    """Wait for the human's ``/approve`` or ``/reject`` comment on the issue.

    The gate polls the issue until a decision appears *after* the latest
    plan comment, so a decision on an older draft is never replayed.
    Approval sends the run to the coder; a rejection sends it back to the
    planner, which revises the plan using the feedback from the rejection.

    Returns:
        The state update recording the decision: ``plan_approved`` plus the
        iteration counter. The conditional edge after this node routes on
        ``plan_approved``.
    """
    settings = get_settings()
    issue_id = state["issue_id"]
    deadline = (
        time.monotonic() + settings.approval_timeout
        if settings.approval_timeout > 0
        else None
    )
    announced = False
    while True:
        context = fetch_issue_context(issue_id)
        if context.plan is None:
            raise RuntimeError(
                f"issue {issue_id} carries no plan to approve — the planner "
                "must post a plan comment first"
            )
        decision = context.decision_after(context.plan_comment_id)
        if decision is not None:
            if decision["action"] == "approve":
                console.print(
                    f"[bold green]✔ plan approved[/bold green] "
                    f"(issue {issue_id}) — starting implementation"
                )
                return {"plan_approved": True, "iteration": 0}
            console.print(
                f"[bold yellow]✘ plan rejected[/bold yellow] (issue {issue_id}) "
                "— the Planner is revising it"
            )
            return {
                "plan_approved": False,
                "iteration": state.get("iteration", 0) + 1,
            }
        if not announced:
            console.print(
                f"[bold magenta]⏸ waiting for plan approval[/bold magenta] — "
                f"comment `/approve` or `/reject <reason>` on issue {issue_id}"
            )
            announced = True
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(
                f"no plan decision on issue {issue_id} after "
                f"{settings.approval_timeout:.0f}s"
            )
        time.sleep(settings.approval_poll_interval)


def _remote_node(name: str):
    """Wrap a remote node service (one container per node) as a graph node."""

    def node(state: TeamState) -> dict:
        return call_node(name, dict(state))

    node.__name__ = f"{name}_remote"
    return node

def build_graph(
    checkpointer: BaseCheckpointSaver,
):
    """Compile the agent graph; agent nodes run in their own containers over
    HTTP (see rpc.py). Node functions follow the naming convention
    (agents/<name>.py defines <name>()).

    The approve_plan gate sits between planner and coder; it waits for the
    human's ``/approve`` / ``/reject`` comment on the run's GitHub issue
    before any coding starts. The gate records the decision in
    ``plan_approved`` and a conditional edge routes on it, so the branch is
    visible in the compiled graph and survives a restart.

    A checkpointer is required for state inspection and for resuming a run
    after a restart. Callers that need the state to outlive the process
    inject a persistent one (see checkpoint.py); callers that only need a
    temporary graph can inject an in-memory saver explicitly.
    """
    graph = StateGraph(TeamState)
    graph.add_node("planner", _remote_node("planner"))
    graph.add_node("coder", _remote_node("coder"))
    graph.add_node("tester", _remote_node("tester"))
    graph.add_node("reviewer", _remote_node("reviewer"))
    graph.add_node("approve_plan", approve_plan)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "approve_plan")
    graph.add_edge("coder", "tester")
    graph.add_edge("tester", "reviewer")
    graph.add_conditional_edges(
        "reviewer", _route_after_review, {"coder": "coder", END: END}
    )
    graph.add_conditional_edges(
        "approve_plan",
        _route_after_plan_approval,
        {"planner": "planner", "coder": "coder"},
    )

    if checkpointer is None:
        raise ValueError("checkpointer is required")
    return graph.compile(checkpointer=checkpointer)
