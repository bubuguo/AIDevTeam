"""Unit tests for the persistent (SQLite) checkpointer."""

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from aidevteam.checkpoint import (
    checkpoint_db_path,
    list_threads,
    persistent_checkpointer,
)
from aidevteam.config import get_settings
from aidevteam.graph import build_graph
from aidevteam.state import Review, TeamState


def _tiny_graph(saver):
    """A planner → approve_plan graph that mimics how the real pipeline pauses.

    The real run pauses at the approval gate via ``interrupt()``: the pending
    interrupt is persisted in the checkpoint and the process can exit. A
    later resume supplies the human's decision with ``Command(resume=...)``.
    """
    graph = StateGraph(TeamState)

    def planner(state):
        # The real planner returns {} (the plan is transient, posted to the
        # GitHub issue). A review verdict is written here anyway so the
        # round-trip of a pydantic model through the database is exercised.
        return {"review": Review(approved=False, feedback=["fix the loop"])}

    def approve_plan(state):
        decision = interrupt({"issue_id": state["issue_id"], "plan_comment_id": 1})
        return {"plan_approved": decision.get("action") == "approve"}

    graph.add_node("planner", planner)
    graph.add_node("approve_plan", approve_plan)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "approve_plan")
    graph.add_edge("approve_plan", END)
    return graph.compile(checkpointer=saver)


def test_checkpoint_db_path_creates_the_parent_directory(tmp_path):
    path = checkpoint_db_path(tmp_path / "nested" / "ckpt.sqlite")
    assert path.parent.is_dir()
    assert path.name == "ckpt.sqlite"


def test_persistent_checkpointer_creates_the_database_file(tmp_path):
    db = tmp_path / "ckpt.sqlite"
    with persistent_checkpointer(db) as saver:
        assert db.is_file()
        assert saver.is_setup
    assert db.is_file()


def test_state_survives_closing_and_reopening_the_checkpointer(tmp_path):
    """The whole point of persistent storage: the state outlives the process."""
    db = tmp_path / "ckpt.sqlite"
    config = {"configurable": {"thread_id": "12345"}}

    with persistent_checkpointer(db) as saver:
        graph = _tiny_graph(saver)
        result = graph.invoke({"issue_id": 12345}, config)
        assert result.get("__interrupt__"), "the gate must pause the run"

    # A brand-new connection, as a fresh process would open.
    with persistent_checkpointer(db) as saver:
        graph = _tiny_graph(saver)
        snapshot = graph.get_state(config)
        assert snapshot.values["issue_id"] == 12345
        assert snapshot.next == ("approve_plan",)


def test_pydantic_models_round_trip_through_the_database(tmp_path):
    """The msgpack allowlist is what lets pydantic models survive a reload."""
    db = tmp_path / "ckpt.sqlite"
    config = {"configurable": {"thread_id": "12345"}}

    with persistent_checkpointer(db) as saver:
        graph = _tiny_graph(saver)
        result = graph.invoke({"issue_id": 12345}, config)
        assert result.get("__interrupt__"), "the gate must pause the run"

    with persistent_checkpointer(db) as saver:
        graph = _tiny_graph(saver)
        snapshot = graph.get_state(config)
        # The review verdict comes back as a Review model, not a plain dict.
        assert snapshot.values["review"] == Review(
            approved=False, feedback=["fix the loop"]
        )


def test_a_paused_run_can_be_resumed_from_a_reopened_checkpointer(tmp_path):
    db = tmp_path / "ckpt.sqlite"
    config = {"configurable": {"thread_id": "12345"}}

    with persistent_checkpointer(db) as saver:
        graph = _tiny_graph(saver)
        result = graph.invoke({"issue_id": 12345}, config)
        assert result.get("__interrupt__"), "the gate must pause the run"

    with persistent_checkpointer(db) as saver:
        graph = _tiny_graph(saver)
        state = graph.invoke(Command(resume={"action": "approve"}), config)
        assert state["plan_approved"] is True
        assert graph.get_state(config).next == ()


def test_list_threads_reports_stored_threads_newest_first(tmp_path):
    db = tmp_path / "ckpt.sqlite"
    with persistent_checkpointer(db) as saver:
        graph = _tiny_graph(saver)
        # Each invoke pauses at the gate — the checkpoint (and thread) is
        # stored either way.
        graph.invoke(
            {"issue_id": 100},
            {"configurable": {"thread_id": "100"}},
        )
        graph.invoke(
            {"issue_id": 200},
            {"configurable": {"thread_id": "200"}},
        )
        threads = list_threads(saver)

    assert [thread_id for thread_id, _ in threads] == ["200", "100"]
    assert all(timestamp for _, timestamp in threads)


def test_list_threads_reads_only_the_latest_checkpoint_of_each_thread(tmp_path):
    """Listing the runs must not scale with the number of checkpoints stored."""
    db = tmp_path / "ckpt.sqlite"
    config = {"configurable": {"thread_id": "12345"}}
    with persistent_checkpointer(db) as saver:
        graph = _tiny_graph(saver)
        result = graph.invoke({"issue_id": 12345}, config)
        assert result.get("__interrupt__"), "the gate must pause the run"
        graph.invoke(Command(resume={"action": "approve"}), config)

        rows = saver.conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        assert rows > 1  # the run really did write several checkpoints

        read: list[str] = []
        original = saver.get_tuple

        def counting_get_tuple(cfg):
            read.append(str(cfg["configurable"]["thread_id"]))
            return original(cfg)

        saver.get_tuple = counting_get_tuple
        threads = list_threads(saver)

    # One read per thread — not one per checkpoint.
    assert read == ["12345"]
    assert [thread_id for thread_id, _ in threads] == ["12345"]


def test_checkpoint_db_path_falls_back_to_the_default_when_blank(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHECKPOINT_DB", "   ")
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.example")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    get_settings.cache_clear()
    try:
        path = checkpoint_db_path()
    finally:
        get_settings.cache_clear()
    assert path.name == "checkpoints.db"
    assert path.parent.is_dir()
    # The default is relative, so it resolves against the working directory.
    assert path.resolve() == (tmp_path / "state" / "checkpoints.db").resolve()


def test_build_graph_uses_the_injected_checkpointer(tmp_path):
    with persistent_checkpointer(tmp_path / "ckpt.sqlite") as saver:
        graph = build_graph(checkpointer=saver)
        assert graph.checkpointer is saver
        assert isinstance(graph.checkpointer, SqliteSaver)


def test_build_graph_requires_an_explicit_checkpointer():
    """The graph API does not silently create a saver for callers."""
    with pytest.raises(TypeError):
        build_graph()
