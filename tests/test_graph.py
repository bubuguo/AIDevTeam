"""Unit tests for graph construction (containerized execution)."""

from aidevteam.graph import _remote_node, build_graph

# The agents wired into build_graph (explicit add_node calls).
WIRED_AGENTS = ("planner", "coder", "tester", "reviewer")


def test_graph_wires_all_discovered_agents():
    graph = build_graph().get_graph()
    assert set(WIRED_AGENTS) <= set(graph.nodes)


def test_nodes_are_rpc_wrappers():
    for name in WIRED_AGENTS:
        assert _remote_node(name).__name__ == f"{name}_remote"
    graph = build_graph().get_graph()
    assert set(WIRED_AGENTS) <= set(graph.nodes)
