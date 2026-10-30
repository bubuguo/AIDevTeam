"""Unit tests for the persistent (SQLite) checkpointer."""

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from aidevteam.checkpoint import checkpoint_db_path, list_threads, persistent_checkpointer
from aidevteam.config import get_settings
from aidevteam.graph import build_graph
from aidevteam.state import Plan, TeamState


def _tiny_graph(saver):
    """A two-node graph that pauses on an interrupt, like the real pipeline."""
    graph = StateGraph(TeamState)

    def planner(state):
        return {"plan": Plan(app_name="demo", summary="demo app", files=[])}

    def gate(state):
        answer = interrupt({"plan": state["plan"].model_dump(), "revision": 0})
        return {"plan_approved": bool(answer.get("approved"))}

    graph.add_node("planner", planner)
    graph.add_node("approve_plan", gate)
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
    config = {"configurable": {"thread_id": "run-1"}}

    with persistent_checkpointer(db) as saver:
        graph = _tiny_graph(saver)
        graph.invoke({"task": "t", "run_id": "run-1"}, config)

    # A brand-new connection, as a fresh process would open.
    with persistent_checkpointer(db) as saver:
        graph = _tiny_graph(saver)
        snapshot = graph.get_state(config)
        assert snapshot.values["task"] == "t"
        assert snapshot.values["plan"].app_name == "demo"
        assert snapshot.next == ("approve_plan",)
        assert snapshot.interrupts[0].value["plan"]["app_name"] == "demo"


def test_pydantic_models_round_trip_through_the_database(tmp_path):
    """The msgpack allowlist is what makes the Plan model survive a reload."""
    db = tmp_path / "ckpt.sqlite"
    config = {"configurable": {"thread_id": "run-1"}}

    with persistent_checkpointer(db) as saver:
        graph = _tiny_graph(saver)
        graph.invoke({"task": "t", "run_id": "run-1"}, config)

    with persistent_checkpointer(db) as saver:
        graph = _tiny_graph(saver)
        plan = graph.get_state(config).values["plan"]
        assert isinstance(plan, Plan)
        assert plan.app_name == "demo"


def test_a_paused_run_can_be_resumed_from_a_reopened_checkpointer(tmp_path):
    db = tmp_path / "ckpt.sqlite"
    config = {"configurable": {"thread_id": "run-1"}}

    with persistent_checkpointer(db) as saver:
        graph = _tiny_graph(saver)
        graph.invoke({"task": "t", "run_id": "run-1"}, config)

    with persistent_checkpointer(db) as saver:
        graph = _tiny_graph(saver)
        state = graph.invoke(Command(resume={"approved": True}), config)
        assert state["plan_approved"] is True
        assert graph.get_state(config).next == ()


def test_list_threads_reports_stored_threads_newest_first(tmp_path):
    db = tmp_path / "ckpt.sqlite"
    with persistent_checkpointer(db) as saver:
        graph = _tiny_graph(saver)
        graph.invoke(
            {"task": "t", "run_id": "older"},
            {"configurable": {"thread_id": "older"}},
        )
        graph.invoke(
            {"task": "t", "run_id": "newer"},
            {"configurable": {"thread_id": "newer"}},
        )
        threads = list_threads(saver)

    assert [thread_id for thread_id, _ in threads] == ["newer", "older"]
    assert all(timestamp for _, timestamp in threads)


def test_list_threads_reads_only_the_latest_checkpoint_of_each_thread(tmp_path):
    """Listing the runs must not scale with the number of checkpoints stored."""
    db = tmp_path / "ckpt.sqlite"
    config = {"configurable": {"thread_id": "run-1"}}
    with persistent_checkpointer(db) as saver:
        graph = _tiny_graph(saver)
        graph.invoke({"task": "t", "run_id": "run-1"}, config)
        graph.invoke(Command(resume={"approved": True}), config)

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
    assert read == ["run-1"]
    assert [thread_id for thread_id, _ in threads] == ["run-1"]


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
