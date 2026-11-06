"""Unit tests for the node RPC layer (serialization + async job contract)."""

import httpx
import pytest

from aidevteam.rpc import call_node, decode_update, encode_state
from aidevteam.state import CodeResult, FileSpec, GeneratedFile, Plan


def _sample_state() -> dict:
    return {
        "task": "build a csv tool",
        "plan": Plan(
            app_name="csvtool",
            summary="converts csv",
            cli_commands=["app.py convert in.csv out.json"],
            files=[FileSpec(path="app.py", description="entry point")],
        ),
        "iteration": 2,
        "feedback": ["fix the loop"],
        "approved": False,
    }


def test_encode_state_converts_models_to_dicts():
    encoded = encode_state(_sample_state())
    assert isinstance(encoded["plan"], dict)
    assert encoded["plan"]["app_name"] == "csvtool"
    # Plain values pass through untouched.
    assert encoded["task"] == "build a csv tool"
    assert encoded["iteration"] == 2


def test_decode_update_restores_models():
    decoded = decode_update(encode_state(_sample_state()))
    assert isinstance(decoded["plan"], Plan)
    assert decoded["plan"] == _sample_state()["plan"]
    assert decoded["feedback"] == ["fix the loop"]


def test_code_result_round_trip():
    code = CodeResult(files=[GeneratedFile(path="app.py", content="print('hi')")], notes="n")
    decoded = decode_update({"code": code.model_dump()})
    assert decoded["code"] == code


def _fake_response(method: str, url: str, body: dict) -> httpx.Response:
    """Build a standalone httpx.Response that supports raise_for_status()."""
    return httpx.Response(200 if method == "GET" else 202, json=body, request=httpx.Request(method, url))


def test_call_node_submits_and_polls(monkeypatch):
    """call_node should POST /run, then poll /jobs/{id} until done."""
    calls = {"post": 0, "get": 0}

    def fake_post(url, json=None, timeout=None):
        calls["post"] += 1
        assert url.endswith("planner:8000/run")
        return _fake_response("POST", url, {"job_id": "job-1"})

    def fake_get(url, timeout=None):
        calls["get"] += 1
        if calls["get"] == 1:
            return _fake_response("GET", url, {"status": "running", "result": None})
        result = encode_state({"plan": Plan(app_name="x", summary="s", files=[])})
        return _fake_response("GET", url, {"status": "done", "result": result})

    monkeypatch.setattr("aidevteam.rpc.httpx.post", fake_post)
    monkeypatch.setattr("aidevteam.rpc.httpx.get", fake_get)
    monkeypatch.setattr("aidevteam.rpc.time.sleep", lambda _s: None)

    update = call_node("planner", {"task": "t"})
    assert calls["post"] == 1 and calls["get"] == 2
    assert isinstance(update["plan"], Plan)
    assert update["plan"].app_name == "x"


def test_call_node_retries_connection_errors_then_gives_up(monkeypatch):
    """Connection errors during submit are retried within the grace window."""
    monkeypatch.setattr(
        "aidevteam.rpc.httpx.post",
        lambda *a, **kw: (_ for _ in ()).throw(httpx.ConnectError("refused")),
    )
    monkeypatch.setattr("aidevteam.rpc.time.sleep", lambda _s: None)

    # Fake clock that advances 10s per call, so the 60s grace window expires
    # after a few retries instead of looping forever.
    class _Clock:
        value = 0.0

        def __call__(self):
            self.value += 10.0
            return self.value

    monkeypatch.setattr("aidevteam.rpc.time.monotonic", _Clock())

    with pytest.raises(RuntimeError, match="unreachable"):
        call_node("coder", {"task": "t"})


def test_call_node_raises_on_job_error_without_reexecution(monkeypatch):
    """A failed job must raise immediately, pointing at the node's logs."""

    def fake_post(url, json=None, timeout=None):
        return _fake_response("POST", url, {"job_id": "job-9"})

    def fake_get(url, timeout=None):
        return _fake_response("GET", url, {"status": "error", "error": "ValueError: bad plan"})

    monkeypatch.setattr("aidevteam.rpc.httpx.post", fake_post)
    monkeypatch.setattr("aidevteam.rpc.httpx.get", fake_get)
    monkeypatch.setattr("aidevteam.rpc.time.sleep", lambda _s: None)

    with pytest.raises(RuntimeError, match=r"bad plan.*podman logs aidevteam_reviewer_1"):
        call_node("reviewer", {"task": "t"})


def test_call_node_poll_tolerates_server_errors(monkeypatch):
    """A 5xx blip while polling (proxy hiccup, node restart) is retried
    until the deadline, not fatal."""

    def fake_post(url, json=None, timeout=None):
        return _fake_response("POST", url, {"job_id": "job-5"})

    calls = {"get": 0}

    def fake_get(url, timeout=None):
        calls["get"] += 1
        if calls["get"] == 1:
            return httpx.Response(503, request=httpx.Request("GET", url))
        return _fake_response("GET", url, {"status": "done", "result": {"iteration": 1}})

    monkeypatch.setattr("aidevteam.rpc.httpx.post", fake_post)
    monkeypatch.setattr("aidevteam.rpc.httpx.get", fake_get)
    monkeypatch.setattr("aidevteam.rpc.time.sleep", lambda _s: None)

    update = call_node("coder", {"issue_id": 1})
    assert calls["get"] == 2
    assert update["iteration"] == 1


def test_call_node_poll_fails_fast_on_client_error(monkeypatch):
    """A 4xx while polling means the job is really gone (e.g. the node lost
    its job store) — raise immediately instead of waiting out the deadline."""

    def fake_post(url, json=None, timeout=None):
        return _fake_response("POST", url, {"job_id": "job-6"})

    def fake_get(url, timeout=None):
        return httpx.Response(404, request=httpx.Request("GET", url))

    monkeypatch.setattr("aidevteam.rpc.httpx.post", fake_post)
    monkeypatch.setattr("aidevteam.rpc.httpx.get", fake_get)
    monkeypatch.setattr("aidevteam.rpc.time.sleep", lambda _s: None)

    with pytest.raises(RuntimeError, match="lost while polling"):
        call_node("coder", {"issue_id": 1})


def test_call_node_retries_server_errors_on_submit(monkeypatch):
    """A 5xx while submitting (node still booting behind a proxy) is retried
    within the grace window instead of escaping as a raw httpx error."""
    calls = {"post": 0}

    def fake_post(url, json=None, timeout=None):
        calls["post"] += 1
        if calls["post"] == 1:
            return httpx.Response(503, request=httpx.Request("POST", url))
        return _fake_response("POST", url, {"job_id": "job-7"})

    def fake_get(url, timeout=None):
        return _fake_response("GET", url, {"status": "done", "result": {}})

    monkeypatch.setattr("aidevteam.rpc.httpx.post", fake_post)
    monkeypatch.setattr("aidevteam.rpc.httpx.get", fake_get)
    monkeypatch.setattr("aidevteam.rpc.time.sleep", lambda _s: None)

    update = call_node("coder", {"issue_id": 1})
    assert calls["post"] == 2
    assert update == {}


def test_call_node_fails_fast_on_client_error_on_submit(monkeypatch):
    """A 4xx on submit means the request itself is wrong — no retry, and the
    error is a RuntimeError the CLI can report."""

    def fake_post(url, json=None, timeout=None):
        return httpx.Response(422, request=httpx.Request("POST", url))

    monkeypatch.setattr("aidevteam.rpc.httpx.post", fake_post)

    with pytest.raises(RuntimeError, match="rejected the job submission"):
        call_node("coder", {"issue_id": 1})


def test_call_node_reports_a_missing_job_id(monkeypatch):
    """A 200 without a job id is a hard error, not a silent hang."""

    def fake_post(url, json=None, timeout=None):
        return _fake_response("POST", url, {"unexpected": "shape"})

    monkeypatch.setattr("aidevteam.rpc.httpx.post", fake_post)

    with pytest.raises(RuntimeError, match="no job id"):
        call_node("coder", {"issue_id": 1})


def test_call_node_poll_tolerates_non_json_responses(monkeypatch):
    """A proxy answering with an HTML error page must not crash the poll —
    it is retried like any other transient failure."""

    def fake_post(url, json=None, timeout=None):
        return _fake_response("POST", url, {"job_id": "job-8"})

    calls = {"get": 0}

    def fake_get(url, timeout=None):
        calls["get"] += 1
        if calls["get"] == 1:
            return httpx.Response(
                200,
                text="<html>gateway error</html>",
                request=httpx.Request("GET", url),
            )
        return _fake_response("GET", url, {"status": "done", "result": {"iteration": 3}})

    monkeypatch.setattr("aidevteam.rpc.httpx.post", fake_post)
    monkeypatch.setattr("aidevteam.rpc.httpx.get", fake_get)
    monkeypatch.setattr("aidevteam.rpc.time.sleep", lambda _s: None)

    update = call_node("coder", {"issue_id": 1})
    assert calls["get"] == 2
    assert update["iteration"] == 3


def test_call_node_fails_fast_when_the_job_is_already_running(monkeypatch):
    """A 409 means the same job is still executing on the node — never
    re-execute it; surface the conflict instead."""

    def fake_post(url, json=None, timeout=None):
        return httpx.Response(409, request=httpx.Request("POST", url))

    monkeypatch.setattr("aidevteam.rpc.httpx.post", fake_post)

    with pytest.raises(RuntimeError, match="already running"):
        call_node("coder", {"issue_id": 1})
