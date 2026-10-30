"""Unit tests for graph construction (containerized execution)."""

import pytest
from langgraph.checkpoint.memory import MemorySaver

from aidevteam.graph import (
    _approve_plan_node,
    _remote_node,
    _route_after_plan_approval,
    _route_after_review,
    approve_plan,
    build_graph,
)

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


def test_graph_without_plan_approval_passes_through_the_gate(monkeypatch):
    """Disabled gate: same topology, but the node never interrupts."""
    graph = build_graph(plan_approval=False, checkpointer=MemorySaver()).get_graph()
    assert "approve_plan" in graph.nodes
    edges = {(edge.source, edge.target) for edge in graph.edges}
    assert ("planner", "approve_plan") in edges

    # The pass-through must not pause the run: interrupt() must stay uncalled.
    def boom(payload):
        raise AssertionError("interrupt() called with the gate disabled")

    monkeypatch.setattr("aidevteam.graph.interrupt", boom)
    node = _approve_plan_node(False)
    assert node({"plan": None}) == {"plan_approved": True}


def test_graph_always_has_checkpointer():
    """The gate needs it for interrupt()/resume; plain runs get inspectable
    state (get_state / state history) for free."""
    assert build_graph(checkpointer=MemorySaver()).checkpointer is not None
    assert (
        build_graph(plan_approval=False, checkpointer=MemorySaver()).checkpointer
        is not None
    )


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


def test_approve_plan_accumulates_rejection_feedback(monkeypatch):
    monkeypatch.setattr(
        "aidevteam.graph.interrupt",
        lambda payload: {"approved": False, "feedback": "too many files"},
    )
    update = approve_plan(
        {"plan": None, "plan_feedback": ["smaller scope"], "iteration": 1}
    )
    assert update["plan_approved"] is False
    assert update["plan_feedback"] == ["smaller scope", "too many files"]
    assert update["iteration"] == 2


def test_approve_plan_approval_resets_iteration(monkeypatch):
    """Approval resets iteration so plan rejections never eat the coder's
    max_iterations budget; job-id uniqueness comes from the rejection bumps."""
    monkeypatch.setattr(
        "aidevteam.graph.interrupt",
        lambda payload: {"approved": True},
    )
    update = approve_plan(
        {"plan": None, "plan_feedback": ["smaller scope"], "iteration": 2}
    )
    assert update["plan_approved"] is True
    assert update["plan_feedback"] == ["smaller scope"]
    assert update["iteration"] == 0


def test_approve_plan_interrupt_payload_contains_plan(monkeypatch):
    captured: dict = {}

    def fake_interrupt(payload):
        captured.update(payload)
        return {"approved": True}

    monkeypatch.setattr("aidevteam.graph.interrupt", fake_interrupt)
    approve_plan({"plan": {"app_name": "x"}, "iteration": 0})
    assert captured["plan"] == {"app_name": "x"}
    assert captured["revision"] == 0


def test_approve_plan_fails_closed_on_malformed_resume(monkeypatch):
    """A non-dict resume must count as a rejection, never skip the gate."""
    monkeypatch.setattr("aidevteam.graph.interrupt", lambda payload: True)
    update = approve_plan({"plan": None, "iteration": 0})
    assert update["plan_approved"] is False
    assert update["plan_feedback"] == []
    assert update["iteration"] == 1


def test_approve_plan_tolerates_none_resume(monkeypatch):
    monkeypatch.setattr("aidevteam.graph.interrupt", lambda payload: None)
    update = approve_plan({"plan": None, "iteration": 0})
    assert update["plan_approved"] is False
    assert update["iteration"] == 1
