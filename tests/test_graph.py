"""Unit tests for graph construction (containerized execution)."""

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from aidevteam.graph import (
    _remote_node,
    _route_after_plan_approval,
    _route_after_review,
    approve_plan,
    build_graph,
)
from aidevteam.issuecontext import IssueContext
from aidevteam.state import Plan, TeamState

# The agents wired into build_graph (explicit add_node calls).
WIRED_AGENTS = ("planner", "coder", "tester", "reviewer")


def test_graph_wires_all_discovered_agents():
    graph = build_graph(checkpointer=MemorySaver()).get_graph()
    assert set(WIRED_AGENTS) <= set(graph.nodes)


def test_nodes_are_rpc_wrappers():
    for name in WIRED_AGENTS:
        assert _remote_node(name).__name__ == f"{name}_remote"
    graph = build_graph(checkpointer=MemorySaver()).get_graph()
    assert set(WIRED_AGENTS) <= set(graph.nodes)


def test_plan_approval_gate_is_wired_by_default():
    """The planner must route through the human gate before any coding starts."""
    graph = build_graph(checkpointer=MemorySaver()).get_graph()
    assert "approve_plan" in graph.nodes
    edges = {(edge.source, edge.target) for edge in graph.edges}
    assert ("planner", "approve_plan") in edges


def test_graph_always_has_checkpointer():
    """The gate needs it to poll the issue across restarts; plain runs get
    inspectable state (get_state / state history) for free."""
    assert build_graph(checkpointer=MemorySaver()).checkpointer is not None


def test_build_graph_requires_a_checkpointer():
    with pytest.raises(TypeError):
        build_graph()
    with pytest.raises(ValueError, match="checkpointer is required"):
        build_graph(checkpointer=None)


def test_approved_plan_goes_to_coder():
    assert _route_after_plan_approval({"plan_approved": True}) == "coder"


def test_rejected_plan_goes_back_to_planner():
    assert _route_after_plan_approval({"plan_approved": False}) == "planner"
    assert _route_after_plan_approval({}) == "planner"


def test_review_routing_still_ends_on_approval():
    assert _route_after_review({"approved": True}) == "__end__"
    assert _route_after_review({"approved": False, "iteration": 0}) == "coder"


def _gate_graph():
    """Compile just the approval gate so interrupt()/resume can be exercised."""
    graph = StateGraph(TeamState)
    graph.add_node("approve_plan", approve_plan)
    graph.add_edge(START, "approve_plan")
    graph.add_edge("approve_plan", END)
    return graph.compile(checkpointer=MemorySaver())


def _config(thread_id: str = "12345"):
    return {"configurable": {"thread_id": thread_id}}


def _issue_context(decisions=(), plan_comment_id: int = 1, with_plan: bool = True):
    """An IssueContext as fetch_issue_context would build it from the issue."""
    return IssueContext(
        issue_id=12345,
        task="build a demo",
        plan=Plan(app_name="demo", summary="demo app", files=[]) if with_plan else None,
        plan_comment_id=plan_comment_id,
        decisions=list(decisions),
    )


def test_approve_plan_interrupts_with_the_plan_comment_id(monkeypatch):
    """The gate pauses the run and records where the decision must appear:
    the interrupt payload carries the issue and the latest plan comment."""
    context = _issue_context()
    monkeypatch.setattr(
        "aidevteam.graph.fetch_issue_context", lambda issue_id, tool=None: context
    )
    result = _gate_graph().invoke({"issue_id": 12345, "iteration": 0}, _config())
    interrupts = result["__interrupt__"]
    assert len(interrupts) == 1
    assert interrupts[0].value == {"issue_id": 12345, "plan_comment_id": 1}


def test_approve_plan_resume_approval_resets_iteration(monkeypatch):
    """Resuming with the human's /approve sends the run on with a reset
    iteration counter, so plan rejections never eat the coder's budget."""
    context = _issue_context()
    monkeypatch.setattr(
        "aidevteam.graph.fetch_issue_context", lambda issue_id, tool=None: context
    )
    graph = _gate_graph()
    config = _config()
    graph.invoke({"issue_id": 12345, "iteration": 3}, config)
    state = graph.invoke(Command(resume={"action": "approve"}), config)
    assert state["plan_approved"] is True
    assert state["iteration"] == 0


def test_approve_plan_resume_rejection_bumps_iteration(monkeypatch):
    """Resuming with a /reject sends the run back to the planner and counts
    the revision; the feedback itself lives on the issue, not in the state."""
    context = _issue_context()
    monkeypatch.setattr(
        "aidevteam.graph.fetch_issue_context", lambda issue_id, tool=None: context
    )
    graph = _gate_graph()
    config = _config()
    graph.invoke({"issue_id": 12345, "iteration": 1}, config)
    state = graph.invoke(
        Command(resume={"action": "reject", "feedback": "too many files"}), config
    )
    assert state["plan_approved"] is False
    assert state["iteration"] == 2


def test_approve_plan_fails_closed_without_a_plan(monkeypatch):
    """An issue without a plan comment is an error, never an implicit approval."""
    context = _issue_context(with_plan=False, plan_comment_id=0)
    monkeypatch.setattr(
        "aidevteam.graph.fetch_issue_context", lambda issue_id, tool=None: context
    )
    with pytest.raises(RuntimeError, match="carries no plan"):
        _gate_graph().invoke({"issue_id": 12345}, _config())


def test_approve_plan_fails_closed_on_an_unreadable_plan(monkeypatch):
    """A plan comment whose JSON block is broken must not leave an older plan
    active nor be approved blind — the Planner has to re-post it."""
    context = _issue_context(with_plan=False, plan_comment_id=7)
    monkeypatch.setattr(
        "aidevteam.graph.fetch_issue_context", lambda issue_id, tool=None: context
    )
    with pytest.raises(RuntimeError, match="no readable plan"):
        _gate_graph().invoke({"issue_id": 12345}, _config())


def test_approve_plan_stays_paused_on_an_empty_resume(monkeypatch):
    """Resuming without a decision must not silently reject or approve the
    plan: a falsy resume value re-raises the interrupt, leaving the run
    paused exactly as before."""
    context = _issue_context()
    monkeypatch.setattr(
        "aidevteam.graph.fetch_issue_context", lambda issue_id, tool=None: context
    )
    graph = _gate_graph()
    config = _config()
    graph.invoke({"issue_id": 12345}, config)
    result = graph.invoke(Command(resume={}), config)
    assert result.get("__interrupt__"), "the run must stay paused"
    assert graph.get_state(config).next == ("approve_plan",)
