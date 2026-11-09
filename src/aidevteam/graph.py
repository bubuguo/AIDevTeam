"""LangGraph wiring for the Planner → Coder → Tester → Reviewer pipeline.

START → planner → approve_plan → coder → tester → reviewer ─→ approved → END
                 ↑                    ↑                 │
                 └── plan rejected ───┘                 └── revisions ──┘

The run is driven by its GitHub issue: the task lives in the issue body,
every plan draft is posted as an issue comment, and the human approves or
rejects a plan by commenting ``/approve`` or ``/reject <reason>``.

The approval gate does not poll: after the planner posts a plan comment it
calls ``interrupt()``, which persists a pending interrupt in the checkpoint
and lets the process exit. A later ``resume`` (see cli.py) reads the
decision from the issue and continues the run with ``Command(resume=...)``.

The task, the plan and the feedback are *transient* context. They are read
from GitHub whenever a node needs them and are never written to the
checkpoint, so a checkpoint stays small and a run can be resumed from the
issue alone. The human's verdict is the exception: it is persisted as
``plan_approved`` so the conditional edge after the approval gate can route
on it.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from rich.console import Console

from aidevteam.config import get_settings
from aidevteam.issuecontext import fetch_issue_context
from aidevteam.rpc import call_node
from aidevteam.state import TeamState

console = Console()


def _route_after_review(state: TeamState) -> str:
    """Send approved work to END; otherwise route to coder or tester.

    Test-only revisions also count against max_iterations.
    """
    settings = get_settings()
    if state.get("approved"):
        return END
    if state.get("iteration", 0) >= settings.max_iterations:
        return END
    review = state.get("review")
    target = getattr(review, "revision_target", "code")
    return "tester" if target == "tests" else "coder"


def _route_after_plan_approval(state: TeamState) -> str:
    """Send an approved plan to the coder; otherwise back to the planner.

    The gate records the human's decision in ``plan_approved``, so the
    branch is driven by persisted state rather than by an in-memory
    routing instruction. A missing flag counts as "not approved".
    """
    return "coder" if state.get("plan_approved") else "planner"


def approve_plan(state: TeamState) -> dict:
    """Pause the run until the human decides on the issue.

    The gate interrupts the graph right after the planner has posted a plan
    comment: the interrupt payload (the issue id and the plan comment id) is
    persisted in the checkpoint and the process is free to exit. A later
    ``resume`` reads the decision from the issue and continues the run by
    passing it as ``Command(resume=...)`` — the resume value becomes this
    node's ``decision``.

    The decision must have been posted *after* the latest plan comment; that
    staleness check lives in the resuming CLI (see ``_decision_from_issue``),
    which only supplies decisions read from the interrupt payload's comment
    id. Approval sends the run to the coder; a rejection sends it back to
    the planner, which revises the plan using the feedback on the issue.

    Fails closed: an issue without a plan comment (or with an unreadable
    plan) is an error, never an implicit approval.

    Returns:
        The state update recording the decision: ``plan_approved`` plus the
        iteration counter. The conditional edge after this node routes on
        ``plan_approved``.
    """
    issue_id = state["issue_id"]
    context = fetch_issue_context(issue_id)
    if context.plan_comment_id == 0:
        raise RuntimeError(
            f"issue {issue_id} carries no plan to approve — the planner "
            "must post a plan comment first"
        )
    if context.plan is None:
        raise RuntimeError(
            f"issue {issue_id} has a plan comment but no readable "
            "plan in it — the Planner must re-post the plan"
        )

    decision = interrupt(
        {"issue_id": issue_id, "plan_comment_id": context.plan_comment_id}
    )
    if not isinstance(decision, dict) or not decision:
        raise RuntimeError(
            f"issue {issue_id} was resumed without a plan decision — pass "
            "Command(resume={'action': 'approve'} | {'action': 'reject'})"
        )
    if decision.get("action") == "approve":
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

    The approve_plan gate sits between planner and coder: after the planner
    posts a plan comment it interrupts the run, persisting the wait in the
    checkpoint so the process can exit. A later ``resume`` supplies the
    human's ``/approve`` / ``/reject`` decision (read from the issue) as
    ``Command(resume=...)``; the gate records it in ``plan_approved`` and a
    conditional edge routes on it, so the branch is visible in the compiled
    graph and survives a restart.

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
        "reviewer", _route_after_review, {"coder": "coder", "tester": "tester", END: END}
    )
    graph.add_conditional_edges(
        "approve_plan",
        _route_after_plan_approval,
        {"planner": "planner", "coder": "coder"},
    )

    if checkpointer is None:
        raise ValueError("checkpointer is required")
    return graph.compile(checkpointer=checkpointer)
