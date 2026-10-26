"""LangGraph wiring for the Planner → Coder → Tester → Reviewer pipeline.

Flow:
    START → planner → coder → tester → reviewer ─→ approved → END
                        ↑                 │
                        └──── revisions ──┘   (max N iterations)
"""

from langgraph.graph import END, START, StateGraph

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


def _remote_node(name: str):
    """Wrap a remote node service (one container per node) as a graph node."""

    def node(state: TeamState) -> dict:
        return call_node(name, dict(state))

    node.__name__ = f"{name}_remote"
    return node


def build_graph():
    """Compile the multi-agent graph.

    Every node runs in its own container; the graph drives them over HTTP
    (async job pattern, see rpc.py). Node functions follow the naming
    convention (agents/<name>.py defines <name>()). The wiring below is
    intentionally explicit — workflows are not always linear chains and
    conditional edges may live on any node.
    """
    graph = StateGraph(TeamState)
    graph.add_node("planner", _remote_node("planner"))
    graph.add_node("coder", _remote_node("coder"))
    graph.add_node("tester", _remote_node("tester"))
    graph.add_node("reviewer", _remote_node("reviewer"))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "coder")
    graph.add_edge("coder", "tester")
    graph.add_edge("tester", "reviewer")
    graph.add_conditional_edges(
        "reviewer", _route_after_review, {"coder": "coder", END: END}
    )
    return graph.compile()
