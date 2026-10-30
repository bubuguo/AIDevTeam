"""Unit tests for the CLI commands backed by the persistent checkpointer."""

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Interrupt, StateSnapshot, interrupt
from typer.testing import CliRunner

from aidevteam.checkpoint import persistent_checkpointer
from aidevteam.cli import app
from aidevteam.config import get_settings
from aidevteam.state import Plan, TeamState


@pytest.fixture()
def checkpoint_db(tmp_path, monkeypatch):
    """Point the CLI at a throwaway checkpoint database."""
    db = tmp_path / "checkpoints.sqlite"
    monkeypatch.setenv("CHECKPOINT_DB", str(db))
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.example")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("COLUMNS", "200")  # keep Rich tables on a single line
    get_settings.cache_clear()
    yield db
    get_settings.cache_clear()


def _seed_run(db, thread_id: str, finish: bool = False) -> None:
    """Store a run that waits for a plan approval (or one that already finished)."""
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

    config = {"configurable": {"thread_id": thread_id}}
    with persistent_checkpointer(db) as saver:
        compiled = graph.compile(checkpointer=saver)
        compiled.invoke(
            {"task": "build a demo", "run_id": thread_id, "plan_approval": True},
            config,
        )
        if finish:
            compiled.invoke(Command(resume={"approved": True}), config)


def _seed_crashed_run(db, thread_id: str) -> None:
    """Store a run whose node crashed: paused mid-pipeline, no pending approval."""
    graph = StateGraph(TeamState)

    def crashing(state):
        raise RuntimeError("node crashed")

    graph.add_node("coder", crashing)
    graph.add_edge(START, "coder")
    graph.add_edge("coder", END)

    config = {"configurable": {"thread_id": thread_id}}
    with persistent_checkpointer(db) as saver:
        compiled = graph.compile(checkpointer=saver)
        try:
            compiled.invoke({"task": "build a demo", "run_id": thread_id}, config)
        except RuntimeError:
            pass  # the crash is the point: the run stays paused in the store


def test_list_reports_an_empty_store(checkpoint_db):
    result = CliRunner().invoke(app, ["list"])
    assert result.exit_code == 0
    assert "no runs recorded yet" in result.output


def test_list_lists_a_paused_run(checkpoint_db):
    _seed_run(checkpoint_db, "run-1")
    result = CliRunner().invoke(app, ["list"])
    assert result.exit_code == 0
    assert "run-1" in result.output
    assert "awaiting approval" in result.output
    assert "demo" in result.output


def test_resume_rejects_an_unknown_run(checkpoint_db):
    result = CliRunner().invoke(app, ["resume", "does-not-exist"])
    assert result.exit_code == 1
    assert "unknown run" in result.output


def test_resume_reports_a_finished_run(checkpoint_db):
    _seed_run(checkpoint_db, "run-2", finish=True)
    result = CliRunner().invoke(app, ["resume", "run-2"])
    assert result.exit_code == 0
    assert "already finished" in result.output


def test_resume_approves_a_paused_run(checkpoint_db, monkeypatch):
    _seed_run(checkpoint_db, "run-1")
    captured: dict = {}

    class _FakeGraph:
        def __init__(self, plan_approval: bool = True, checkpointer=None):
            captured["plan_approval"] = plan_approval
            captured["checkpointer"] = checkpointer

        def get_state(self, config):
            return StateSnapshot(
                values={
                    "task": "build a demo",
                    "run_id": "run-1",
                    "plan_approval": True,
                },
                next=("approve_plan",),
                config=config,
                metadata={},
                created_at=None,
                parent_config=None,
                tasks=(),
                interrupts=(
                    Interrupt(
                        value={"plan": {"app_name": "demo"}, "revision": 0}, id="i1"
                    ),
                ),
            )

        def invoke(self, command, config):
            captured["command"] = command
            return {"plan_approved": True}

    monkeypatch.setattr(
        "aidevteam.cli.build_graph", lambda **kwargs: _FakeGraph(**kwargs)
    )
    result = CliRunner().invoke(app, ["resume", "run-1"], input="y\n")
    assert result.exit_code == 0
    assert captured["command"].resume == {"approved": True, "feedback": ""}


def test_resume_continues_a_run_paused_mid_pipeline(checkpoint_db, monkeypatch):
    """A run that died inside a node has no pending approval — it just continues."""
    captured: dict = {}

    class _FakeGraph:
        def __init__(self, plan_approval: bool = True, checkpointer=None):
            pass

        def get_state(self, config):
            return StateSnapshot(
                values={
                    "task": "build a demo",
                    "run_id": "run-1",
                    "plan_approval": True,
                },
                next=("coder",),
                config=config,
                metadata={},
                created_at=None,
                parent_config=None,
                tasks=(),
                interrupts=(),
            )

        def invoke(self, command, config):
            captured["command"] = command
            return {"task": "build a demo", "run_id": "run-1", "approved": True}

    monkeypatch.setattr(
        "aidevteam.cli.build_graph", lambda **kwargs: _FakeGraph(**kwargs)
    )
    result = CliRunner().invoke(app, ["resume", "run-1"])
    assert result.exit_code == 0
    # Resuming without new input re-runs the unfinished node.
    assert captured["command"] is None
    assert "paused mid-pipeline" in result.output


def test_list_shows_a_crashed_run_as_paused(checkpoint_db):
    _seed_crashed_run(checkpoint_db, "run-3")
    result = CliRunner().invoke(app, ["list"])
    assert result.exit_code == 0
    assert "run-3" in result.output
    assert "paused" in result.output


def test_run_stores_the_workflow_in_the_persistent_checkpointer(checkpoint_db, monkeypatch):
    captured: dict = {}

    class _FakeGraph:
        def __init__(self, plan_approval: bool = True, checkpointer=None):
            captured["plan_approval"] = plan_approval
            captured["checkpointer"] = checkpointer

        def invoke(self, payload, config):
            captured["payload"] = payload
            captured["config"] = config
            return {"task": payload["task"], "run_id": payload["run_id"]}

    monkeypatch.setattr(
        "aidevteam.cli.build_graph", lambda **kwargs: _FakeGraph(**kwargs)
    )
    result = CliRunner().invoke(app, ["run", "build a demo"])
    assert result.exit_code == 0
    assert isinstance(captured["checkpointer"], SqliteSaver)
    # The gate mode is recorded so `resume` can rebuild the same topology.
    assert captured["payload"]["plan_approval"] is True
    assert captured["config"]["configurable"]["thread_id"] == captured["payload"]["run_id"]
    assert checkpoint_db.is_file()
