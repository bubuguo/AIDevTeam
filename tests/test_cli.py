"""Unit tests for the CLI commands backed by the persistent checkpointer."""

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, StateSnapshot, interrupt
from typer.testing import CliRunner

from aidevteam.checkpoint import persistent_checkpointer
from aidevteam.cli import app
from aidevteam.config import get_settings
from aidevteam.github_issue import GitHubIssueError
from aidevteam.issuecontext import IssueContext
from aidevteam.state import TeamState


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

    def gate(state):
        # The plan itself is transient (it lives on the GitHub issue); the
        # gate interrupts until the human's decision is supplied on resume,
        # like the real gate.
        decision = interrupt({"issue_id": int(thread_id), "plan_comment_id": 1})
        return {"plan_approved": decision.get("action") == "approve"}

    graph.add_node("approve_plan", gate)
    graph.add_edge(START, "approve_plan")
    graph.add_edge("approve_plan", END)

    config = {"configurable": {"thread_id": thread_id}}
    with persistent_checkpointer(db) as saver:
        compiled = graph.compile(checkpointer=saver)
        # Seed the state with issue_id only; plan is transient
        result = compiled.invoke(
            {"issue_id": int(thread_id)},
            config,
        )
        assert result.get("__interrupt__"), "the gate must pause the run"
        if finish:
            # The decision arrived: the gate records it and the run completes.
            compiled.invoke(Command(resume={"action": "approve"}), config)


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
            compiled.invoke({"issue_id": int(thread_id)}, config)
        except RuntimeError:
            pass  # the crash is the point: the run stays paused in the store


def test_list_reports_an_empty_store(checkpoint_db):
    result = CliRunner().invoke(app, ["list"])
    assert result.exit_code == 0
    assert "no issues recorded yet" in result.output


def test_list_lists_a_paused_run(checkpoint_db):
    _seed_run(checkpoint_db, "12345")
    result = CliRunner().invoke(app, ["list"])
    assert result.exit_code == 0
    assert "12345" in result.output
    assert "awaiting approval" in result.output
    # The plan is transient (it lives on the GitHub issue), so it is not
    # part of the checkpoint and cannot show up in the listing.
    assert "demo" not in result.output


def test_resume_rejects_an_unknown_run(checkpoint_db):
    result = CliRunner().invoke(app, ["resume", "99999"])
    assert result.exit_code == 1
    assert "unknown issue" in result.output


def test_resume_reports_a_finished_run(checkpoint_db):
    _seed_run(checkpoint_db, "12345", finish=True)
    result = CliRunner().invoke(app, ["resume", "12345"])
    assert result.exit_code == 0
    assert "already finished" in result.output


def _interrupting_snapshot(config, issue_id: int = 12345) -> StateSnapshot:
    """A snapshot of a run paused at the approval gate, interrupt pending."""
    from types import SimpleNamespace

    return StateSnapshot(
        values={"issue_id": issue_id},
        next=("approve_plan",),
        config=config,
        metadata={},
        created_at=None,
        parent_config=None,
        tasks=(
            SimpleNamespace(
                interrupts=(
                    SimpleNamespace(
                        value={"issue_id": issue_id, "plan_comment_id": 1}
                    ),
                ),
            ),
        ),
        interrupts=(),
    )


def test_resume_approves_a_paused_run(checkpoint_db, monkeypatch):
    """Resuming a run paused at the gate reads the human's /approve from the
    issue and continues the graph with it."""
    captured: dict = {}

    class _FakeGraph:
        def __init__(self, checkpointer=None):
            captured["checkpointer"] = checkpointer

        def get_state(self, config):
            return _interrupting_snapshot(config)

        def invoke(self, command, config):
            captured["command"] = command
            return {"issue_id": 12345, "plan_approved": True}

    monkeypatch.setattr(
        "aidevteam.cli.build_graph", lambda **kwargs: _FakeGraph(**kwargs)
    )
    monkeypatch.setattr(
        "aidevteam.cli.fetch_issue_context",
        lambda issue_id: IssueContext(
            issue_id=issue_id,
            task="build a demo",
            plan_comment_id=1,
            decisions=[{"comment_id": 2, "action": "approve", "feedback": ""}],
        ),
    )
    result = CliRunner().invoke(app, ["resume", "12345"])
    assert result.exit_code == 0
    # The decision read from the issue is supplied to the graph as the
    # gate's resume value.
    assert captured["command"] == Command(resume={"action": "approve"})


def test_resume_rejects_a_paused_run(checkpoint_db, monkeypatch):
    """A /reject comment on the issue resumes the run into a plan revision."""
    captured: dict = {}

    class _FakeGraph:
        def __init__(self, checkpointer=None):
            pass

        def get_state(self, config):
            return _interrupting_snapshot(config)

        def invoke(self, command, config):
            captured["command"] = command
            return {"issue_id": 12345, "plan_approved": False, "iteration": 1}

    monkeypatch.setattr(
        "aidevteam.cli.build_graph", lambda **kwargs: _FakeGraph(**kwargs)
    )
    monkeypatch.setattr(
        "aidevteam.cli.fetch_issue_context",
        lambda issue_id: IssueContext(
            issue_id=issue_id,
            task="build a demo",
            plan_comment_id=1,
            decisions=[
                {"comment_id": 2, "action": "reject", "feedback": "too many files"}
            ],
        ),
    )
    result = CliRunner().invoke(app, ["resume", "12345"])
    assert result.exit_code == 0
    assert captured["command"] == Command(
        resume={"action": "reject", "feedback": "too many files"}
    )


def test_resume_after_a_reject_reports_the_revised_plan_pause(checkpoint_db, monkeypatch):
    """A rejection loops to the planner, which posts a revised plan and pauses
    at the gate again — resume must report the healthy pause, not an error."""

    class _FakeGraph:
        def __init__(self, checkpointer=None):
            pass

        def get_state(self, config):
            return _interrupting_snapshot(config)

        def invoke(self, command, config):
            # The gate consumed the reject, the planner re-planned, and the
            # gate interrupted again: invoke returns with a pending interrupt.
            return {
                "issue_id": 12345,
                "plan_approved": False,
                "iteration": 1,
                "__interrupt__": (
                    {"value": {"issue_id": 12345, "plan_comment_id": 9}},
                ),
            }

    monkeypatch.setattr(
        "aidevteam.cli.build_graph", lambda **kwargs: _FakeGraph(**kwargs)
    )
    monkeypatch.setattr(
        "aidevteam.cli.fetch_issue_context",
        lambda issue_id: IssueContext(
            issue_id=issue_id,
            task="build a demo",
            plan_comment_id=1,
            decisions=[
                {"comment_id": 2, "action": "reject", "feedback": "too many files"}
            ],
        ),
    )
    result = CliRunner().invoke(app, ["resume", "12345"])
    assert result.exit_code == 0
    assert "waiting for approval" in result.output
    assert "aidevteam resume 12345" in result.output
    assert "pipeline ended early" not in result.output


def test_resume_without_a_decision_reports_and_exits_cleanly(checkpoint_db, monkeypatch):
    """No /approve or /reject after the latest plan yet: nothing to resume —
    a friendly message, exit 0, and the graph is not touched."""
    captured: dict = {}

    class _FakeGraph:
        def __init__(self, checkpointer=None):
            pass

        def get_state(self, config):
            return _interrupting_snapshot(config)

        def invoke(self, command, config):
            captured["invoked"] = True
            return {}

    monkeypatch.setattr(
        "aidevteam.cli.build_graph", lambda **kwargs: _FakeGraph(**kwargs)
    )
    monkeypatch.setattr(
        "aidevteam.cli.fetch_issue_context",
        lambda issue_id: IssueContext(
            issue_id=issue_id,
            task="build a demo",
            plan_comment_id=1,
            decisions=[],  # no decision posted yet
        ),
    )
    result = CliRunner().invoke(app, ["resume", "12345"])
    assert result.exit_code == 0
    assert "nothing to resume" in result.output
    assert "invoked" not in captured


def test_resume_continues_a_run_paused_mid_pipeline(checkpoint_db, monkeypatch):
    """A run that died inside a node has no pending approval — it just continues."""
    captured: dict = {}

    class _FakeGraph:
        def __init__(self, plan_approval: bool = True, checkpointer=None):
            pass

        def get_state(self, config):
            return StateSnapshot(
                values={
                    "issue_id": 12345,
                    "plan_approved": True,
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
            return {"issue_id": 12345, "approved": True}

    monkeypatch.setattr(
        "aidevteam.cli.build_graph", lambda **kwargs: _FakeGraph(**kwargs)
    )
    result = CliRunner().invoke(app, ["resume", "12345"])
    assert result.exit_code == 0
    # Resuming without new input re-runs the unfinished node.
    assert captured["command"] is None
    assert "paused mid-pipeline" in result.output


def test_resume_reports_a_node_failure(checkpoint_db, monkeypatch):
    """A node failure (e.g. a container job crashed) exits with a message,
    not a raw traceback."""

    class _FakeGraph:
        def __init__(self, checkpointer=None):
            pass

        def get_state(self, config):
            return StateSnapshot(
                values={"issue_id": 12345},
                next=("coder",),
                config=config,
                metadata={},
                created_at=None,
                parent_config=None,
                tasks=(),
                interrupts=(),
            )

        def invoke(self, command, config):
            raise RuntimeError("node job coder failed")

    monkeypatch.setattr(
        "aidevteam.cli.build_graph", lambda **kwargs: _FakeGraph(**kwargs)
    )
    result = CliRunner().invoke(app, ["resume", "12345"])
    assert result.exit_code == 1
    assert "resume failed" in result.output
    assert "node job coder failed" in result.output


def _stub_valid_issue(monkeypatch, task: str = "build a demo") -> None:
    """Bypass GitHub: `run` sees an existing, open issue with a task body."""
    monkeypatch.setattr("aidevteam.cli._validate_issue_for_run", lambda issue_id: task)


class _EmptySnapshot:
    """get_state() result for an issue with no checkpoint thread yet."""

    values: dict = {}


def test_run_reports_pipeline_failures(checkpoint_db, monkeypatch):
    """A node failure during `run` exits with a message, not a raw traceback."""
    _stub_valid_issue(monkeypatch)

    class _FakeGraph:
        def __init__(self, checkpointer=None):
            pass

        def get_state(self, config):
            return _EmptySnapshot()

        def invoke(self, payload, config):
            raise RuntimeError("Node 'planner' job 12345-planner-0 failed: boom")

    monkeypatch.setattr(
        "aidevteam.cli.build_graph", lambda **kwargs: _FakeGraph(**kwargs)
    )
    result = CliRunner().invoke(app, ["run", "12345"])
    assert result.exit_code == 1
    assert "pipeline failed" in result.output
    assert "12345-planner-0" in result.output


def test_list_shows_a_crashed_run_as_paused(checkpoint_db):
    _seed_crashed_run(checkpoint_db, "12345")
    result = CliRunner().invoke(app, ["list"])
    assert result.exit_code == 0
    assert "12345" in result.output
    assert "paused" in result.output


def test_run_stores_the_workflow_in_the_persistent_checkpointer(checkpoint_db, monkeypatch):
    captured: dict = {}

    class _FakeGraph:
        def __init__(self, checkpointer=None):
            captured["checkpointer"] = checkpointer

        def get_state(self, config):
            return _EmptySnapshot()

        def invoke(self, payload, config):
            captured["payload"] = payload
            captured["config"] = config
            return {"issue_id": payload["issue_id"]}

    monkeypatch.setattr(
        "aidevteam.cli.build_graph", lambda **kwargs: _FakeGraph(**kwargs)
    )
    _stub_valid_issue(monkeypatch)
    result = CliRunner().invoke(app, ["run", "12345"])
    assert result.exit_code == 0
    assert isinstance(captured["checkpointer"], SqliteSaver)
    assert captured["config"]["configurable"]["thread_id"] == str(captured["payload"]["issue_id"])
    assert checkpoint_db.is_file()


def test_run_rejects_a_closed_issue(checkpoint_db, monkeypatch):
    """`run` only starts pipelines for open issues."""

    def _closed(issue_id: int) -> str:
        raise GitHubIssueError(f"issue {issue_id} is closed, not open")

    monkeypatch.setattr("aidevteam.cli._validate_issue_for_run", _closed)
    result = CliRunner().invoke(app, ["run", "12345"])
    assert result.exit_code == 1
    assert "cannot start issue 12345" in result.output
    assert "closed" in result.output


def test_run_rejects_an_issue_without_a_task(checkpoint_db, monkeypatch):
    """`run` refuses issues whose body carries no task text."""

    def _no_task(issue_id: int) -> str:
        raise GitHubIssueError(f"issue {issue_id} has no task text in its body")

    monkeypatch.setattr("aidevteam.cli._validate_issue_for_run", _no_task)
    result = CliRunner().invoke(app, ["run", "12345"])
    assert result.exit_code == 1
    assert "no task text" in result.output


def test_run_rejects_an_issue_that_already_has_a_run(checkpoint_db, monkeypatch):
    """A second `run` on the same issue exits 1 and points at `resume`."""
    _stub_valid_issue(monkeypatch)

    class _StartedSnapshot:
        values: dict = {"issue_id": 12345}
        tasks: tuple = ()
        next: tuple = ()

    class _FakeGraph:
        def __init__(self, checkpointer=None):
            pass

        def get_state(self, config):
            return _StartedSnapshot()

        def invoke(self, payload, config):  # pragma: no cover - must not run
            raise AssertionError("pipeline must not start twice")

    monkeypatch.setattr(
        "aidevteam.cli.build_graph", lambda **kwargs: _FakeGraph(**kwargs)
    )
    result = CliRunner().invoke(app, ["run", "12345"])
    assert result.exit_code == 1
    assert "already has a run" in result.output
    assert "aidevteam resume 12345" in result.output
