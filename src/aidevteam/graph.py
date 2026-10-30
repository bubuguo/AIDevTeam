"""LangGraph wiring for the Planner → Coder → Tester → Reviewer pipeline.

    START → planner → approve_plan → coder → tester → reviewer ─→ approved → END
               ↑             │       ↑                 │
               └── comment ──┘       └──── revisions ──┘   (max N iterations)                

The gate is always wired in; with plan_approval disabled it passes through
without pausing.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel

from aidevteam.config import get_settings
from aidevteam.rpc import call_node
from aidevteam.state import TeamState


def _route_after_review(state: TeamState) -> str:
    """Send approved work to END; otherwise loop back to the coder."""
    settings = get_settings()
    if state.get("approved"):
        return END
    if state.get("iteration", 0) >= settings.max_iterations:
        return END
    return "coder"


def _route_after_plan_approval(state: TeamState) -> str:
    """Send an approved plan to the coder; a rejected one back to the planner."""
    if state.get("plan_approved"):
        return "coder"
    return "planner"


def approve_plan(state: TeamState) -> dict:
    """Pause the graph until a human approves or rejects the plan.

    Resumed with ``{"approved": bool, "feedback": str}``; fails closed — any
    non-approving or malformed resume counts as a rejection. Rejection
    feedback accumulates in ``plan_feedback`` and bumps ``iteration`` (unique
    job ids per planner run); approval resets ``iteration`` to 0.
    """
    plan = state.get("plan")
    answer = interrupt(
        {
            "plan": plan.model_dump() if isinstance(plan, BaseModel) else plan,
            "revision": state.get("iteration", 0),
        }
    )
    if not isinstance(answer, dict):
        answer = {}
    approved = bool(answer.get("approved"))
    feedback = list(state.get("plan_feedback") or [])
    new_feedback = str(answer.get("feedback", "")).strip()
    if new_feedback:
        feedback.append(new_feedback)
    return {
        "plan_approved": approved,
        "plan_feedback": feedback,
        "iteration": 0 if approved else state.get("iteration", 0) + 1,
    }


def _approve_plan_node(enabled: bool):
    """Wrap the approval gate, bypassing it entirely when disabled.

    The node is always wired into the graph so the topology is identical in
    both modes; with the gate off it returns immediately without calling
    interrupt(), so the run never pauses.
    """

    def node(state: TeamState) -> dict:
        if not enabled:
            return {"plan_approved": True}
        return approve_plan(state)

    node.__name__ = "approve_plan"
    return node


def _remote_node(name: str):
    """Wrap a remote node service (one container per node) as a graph node."""

    def node(state: TeamState) -> dict:
        return call_node(name, dict(state))

    node.__name__ = f"{name}_remote"
    return node


def build_graph(
    checkpointer: BaseCheckpointSaver,
    plan_approval: bool = True,
):
    """Compile the agent graph; agent nodes run in their own containers over
    HTTP (see rpc.py). Node functions follow the naming convention
    (agents/<name>.py defines <name>()).

    The approve_plan gate sits between planner and coder; when
    ``plan_approval`` is false it passes through without pausing.

    A checkpointer is required for interrupt()/resume and for state
    inspection. Callers that need the state to outlive the process inject a
    persistent one (see checkpoint.py); callers that only need a temporary
    graph can inject an in-memory saver explicitly.
    """
    graph = StateGraph(TeamState)
    graph.add_node("planner", _remote_node("planner"))
    graph.add_node("coder", _remote_node("coder"))
    graph.add_node("tester", _remote_node("tester"))
    graph.add_node("reviewer", _remote_node("reviewer"))
    graph.add_node("approve_plan", _approve_plan_node(plan_approval))

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
