"""LangGraph wiring for the Planner → Coder → Tester → Reviewer pipeline.

Flow:
    START → planner → coder → tester → reviewer ─→ approved → END
                        ↑                 │
                        └──── revisions ──┘   (max N iterations)
"""

from langgraph.graph import END, START, StateGraph

from aidevteam.agents import code_node, plan_node, review_node, test_node
from aidevteam.config import get_settings
from aidevteam.state import TeamState


def _route_after_review(state: TeamState) -> str:
    """Send approved work to END; otherwise loop back to the coder."""
    settings = get_settings()
    if state.get("approved"):
        return END
    if state.get("iteration", 0) >= settings.max_iterations:
        return END
    return "coder"


def build_graph():
    """Compile the multi-agent graph."""
    graph = StateGraph(TeamState)
    graph.add_node("planner", plan_node)
    graph.add_node("coder", code_node)
    graph.add_node("tester", test_node)
    graph.add_node("reviewer", review_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "coder")
    graph.add_edge("coder", "tester")
    graph.add_edge("tester", "reviewer")
    graph.add_conditional_edges(
        "reviewer", _route_after_review, {"coder": "coder", END: END}
    )
    return graph.compile()
