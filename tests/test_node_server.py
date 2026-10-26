"""Unit tests for the node HTTP service (async job contract)."""

import time

import pytest
from fastapi.testclient import TestClient

from aidevteam import node_server
from aidevteam.state import CodeResult, GeneratedFile


@pytest.fixture()
def fake_coder(monkeypatch):
    """Replace the real coder agent with a deterministic fake.

    With the naming convention, node functions are resolved from their agent
    module at create_app() time, so patching the module attribute works.
    """

    def coder(state):
        assert state["plan"].app_name == "csvtool"  # model was decoded
        time.sleep(0.05)  # simulate work so the job is observable as "running"
        return {
            "code": CodeResult(files=[GeneratedFile(path="app.py", content="print('hi')")]),
            "iteration": state.get("iteration", 0) + 1,
            "feedback": [],
        }

    monkeypatch.setattr("aidevteam.agents.coder.coder", coder)


def test_unknown_role_exits():
    with pytest.raises(SystemExit):
        node_server.create_app("nonexistent")


def test_health_reports_role():
    client = TestClient(node_server.create_app("planner"))
    assert client.get("/health").json() == {"status": "ok", "role": "planner"}


def test_run_is_async_and_returns_job_id(fake_coder):
    """POST /run must return immediately with a job id, not the result."""
    client = TestClient(node_server.create_app("coder"))
    payload = {
        "task": "t",
        "run_id": "20261024-234247-9f3a",
        "plan": {"app_name": "csvtool", "summary": "s", "files": []},
        "iteration": 1,
    }
    response = client.post("/run", json=payload)
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    # Deterministic id: <run_id>-<role>-<iteration> — greppable across logs.
    assert job_id == "20261024-234247-9f3a-coder-1"

    # Poll the job until it finishes (the fake node takes ~50ms).
    for _ in range(50):
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] != "running":
            break
        time.sleep(0.02)
    assert body["status"] == "done"
    assert body["result"]["code"]["files"][0]["path"] == "app.py"
    assert body["result"]["iteration"] == 2


def test_node_failure_is_reported_via_job_status(monkeypatch):
    def broken(state):
        raise ValueError("bad plan")

    monkeypatch.setattr("aidevteam.agents.planner.planner", broken)
    client = TestClient(node_server.create_app("planner"))
    job_id = client.post(
        "/run", json={"task": "t", "run_id": "r-1", "iteration": 0}
    ).json()["job_id"]
    assert job_id == "r-1-planner-0"

    for _ in range(50):
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] != "running":
            break
        time.sleep(0.02)
    assert body["status"] == "error"
    assert "bad plan" in body["error"]


def test_invalid_state_payload_returns_422():
    client = TestClient(node_server.create_app("planner"))
    response = client.post("/run", json={"plan": {"files": "not-a-list"}})
    assert response.status_code == 422
    assert "Invalid state payload" in response.json()["detail"]


def test_unknown_job_returns_404():
    client = TestClient(node_server.create_app("planner"))
    assert client.get("/jobs/does-not-exist").status_code == 404


def test_run_without_run_id_is_rejected():
    """Every job must be tied to a pipeline run (the orchestrator seeds it)."""
    client = TestClient(node_server.create_app("planner"))
    response = client.post("/run", json={"task": "t"})
    assert response.status_code == 422
    assert "run_id is required" in response.json()["detail"]
