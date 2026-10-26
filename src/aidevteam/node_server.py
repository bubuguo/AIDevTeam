"""Expose a single agent node as an HTTP service (one container per node).

Usage:
    python -m aidevteam.node_server <planner|coder|tester|reviewer> [port]

The service is stateless and asynchronous:

- ``POST /run`` validates a JSON-encoded TeamState slice, starts the node in
  a background thread and immediately returns ``{"job_id": ...}``;
- ``GET /jobs/{id}`` reports ``running`` / ``done`` / ``error`` plus the
  encoded state update once finished.

Node execution time is unpredictable (LLM latency, pytest runs), so no HTTP
connection is held open for the duration of a task — the orchestrator polls
(see ``rpc.call_node``). ``GET /health`` is used by container healthchecks.
"""

import sys
import threading
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException

from aidevteam.agents import load_node
from aidevteam.rpc import decode_update, encode_state

DEFAULT_PORT = 8000


@dataclass
class _Job:
    """Lifecycle of one node execution."""

    status: str = "running"  # running | done | error
    result: dict[str, Any] | None = None
    error: str | None = None


class _JobStore:
    """Thread-safe in-memory job registry with a bounded size."""

    def __init__(self, keep: int = 200) -> None:
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()
        self._keep = keep

    def create(self, job_id: str) -> str:
        with self._lock:
            self._jobs[job_id] = _Job()
            # Bound memory: drop the oldest finished jobs when over capacity.
            while len(self._jobs) > self._keep:
                for job_id_old, job in list(self._jobs.items()):
                    if job.status != "running":
                        del self._jobs[job_id_old]
                        break
                else:
                    break
        return job_id

    def get(self, job_id: str) -> _Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def finish(self, job_id: str, result: dict[str, Any] | None = None, error: str | None = None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.status = "error" if error else "done"
                job.result = result
                job.error = error


def _job_id(role: str, decoded: dict[str, Any]) -> str:
    """Deterministic, greppable job id tied to the run: <run_id>-<role>-<iteration>.

    (run_id, role, incoming iteration) is unique within a run, so no random
    component is needed. The /run endpoint rejects payloads without run_id.
    """
    return f"{decoded['run_id']}-{role}-{decoded.get('iteration', 0)}"


def create_app(role: str) -> FastAPI:
    """Build the FastAPI app serving one node role."""
    try:
        node_func = load_node(role)
    except AttributeError as exc:
        # load_node's message already explains the naming convention.
        raise SystemExit(str(exc)) from exc
    store = _JobStore()

    def _execute(job_id: str, decoded: dict[str, Any]) -> None:
        try:
            update = node_func(decoded)
            store.finish(job_id, result=encode_state(update))
        except Exception as exc:  # surface the failure through the job status
            store.finish(job_id, error=f"{type(exc).__name__}: {exc}")

    app = FastAPI(title=f"aidevteam-{role}")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "role": role}

    @app.post("/run", status_code=202)
    def run(state: dict[str, Any]) -> dict[str, Any]:
        try:
            decoded = decode_update(state)
        except Exception as exc:
            raise HTTPException(
                status_code=422, detail=f"Invalid state payload: {exc}"
            ) from exc
        if not decoded.get("run_id"):
            # Every job is tied to a pipeline run; the orchestrator always
            # seeds run_id into the initial state.
            raise HTTPException(status_code=422, detail="run_id is required")
        job_id = _job_id(role, decoded)
        store.create(job_id)
        threading.Thread(target=_execute, args=(job_id, decoded), daemon=True).start()
        return {"job_id": job_id}

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str) -> dict[str, Any]:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
        return {"status": job.status, "result": job.result, "error": job.error}

    return app


def main() -> None:
    args = sys.argv[1:]
    role = args[0] if args else ""
    port = int(args[1]) if len(args) > 1 else DEFAULT_PORT
    uvicorn.run(create_app(role), host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
