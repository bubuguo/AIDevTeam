"""State serialization and HTTP transport for remote node execution.

Each node (planner/coder/tester/reviewer) runs as an HTTP service in its own
container (see node_server.py); this module is the shared contract:
encode_state/decode_update convert TeamState slices between pydantic models
and JSON-safe dicts, and call_node submits a state and polls until done.
"""

import time
from typing import Any, get_type_hints

import httpx
from pydantic import BaseModel

from aidevteam.config import get_settings
from aidevteam.state import Plan, TeamState

# Model-typed TeamState fields, derived from the state definition so new
# model fields work over RPC without touching this module.
_STATE_MODELS: dict[str, type[BaseModel]] = {
    name: ann
    for name, ann in get_type_hints(TeamState).items()
    if isinstance(ann, type) and issubclass(ann, BaseModel)
}

# The plan is transient context — it is never checkpointed — but it still
# crosses the wire as a pydantic model, so it needs a decoder too.
_AGENT_MODELS: dict[str, type[BaseModel]] = {**_STATE_MODELS, "plan": Plan}


def encode_state(state: dict[str, Any]) -> dict[str, Any]:
    """Convert a state dict (pydantic model instances included) to JSON-safe values."""
    # Every pydantic model must become JSON-safe, whether or not it is a
    # known state model.
    return {
        key: value.model_dump() if isinstance(value, BaseModel) else value
        for key, value in state.items()
    }


def decode_update(update: dict[str, Any]) -> dict[str, Any]:
    """Convert a JSON state update back into a dict with pydantic model instances."""
    decoded: dict[str, Any] = {}
    for key, value in update.items():
        model = _AGENT_MODELS.get(key)
        if model is not None and value is not None:
            decoded[key] = model.model_validate(value)
        else:
            decoded[key] = value
    return decoded


def call_node(role: str, state: dict[str, Any]) -> dict[str, Any]:
    """Submit state to ``role``'s node service and poll until it finishes.

    Submit retries connection errors for ``node_connect_grace`` (containers
    still starting up); a failed job raises immediately — nodes are never
    silently re-executed, as their side effects are not idempotent.
    """
    settings = get_settings()
    base = settings.node_url(role).rstrip("/")
    payload = encode_state(state)
    submit_deadline = time.monotonic() + settings.node_connect_grace

    # Submit the job, retrying while the container is still starting up.
    # Transport errors and 5xx (node still booting, proxy hiccup) are
    # retried within the grace window; a 4xx means the request itself is
    # wrong — retrying cannot help, so fail fast.
    while True:
        try:
            response = httpx.post(f"{base}/run", json=payload, timeout=30.0)
            response.raise_for_status()
            try:
                job_id = response.json()["job_id"]
            except (ValueError, KeyError) as exc:
                raise RuntimeError(
                    f"Node {role!r} returned no job id from {base}/run "
                    f"(HTTP {response.status_code})"
                ) from exc
            break
        except httpx.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 409:
                # The node refused a duplicate submission: the same job is
                # still running there. Never re-execute it (side effects are
                # not idempotent) — surface the conflict instead.
                raise RuntimeError(
                    f"Node {role!r} job is already running at {base} — a "
                    "previous attempt may still be executing"
                ) from exc
            if status is not None and 400 <= status < 500:
                raise RuntimeError(
                    f"Node service {role!r} rejected the job submission "
                    f"(HTTP {status}) at {base}/run: {exc}"
                ) from exc
            if time.monotonic() >= submit_deadline:
                raise RuntimeError(
                    f"Node service {role!r} unreachable at {base} "
                    f"(waited {settings.node_connect_grace:.0f}s for startup): {exc}"
                ) from exc
            time.sleep(1.0)

    # Poll until the job finishes; execution time is unbounded by design.
    poll_deadline = time.monotonic() + settings.node_max_wait
    while True:
        try:
            response = httpx.get(f"{base}/jobs/{job_id}", timeout=30.0)
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            # Transport blips, 5xx (proxy hiccups, node restarts) and
            # non-JSON bodies (e.g. an HTML error page) are transient —
            # keep waiting until the deadline. A 4xx means the job is
            # really gone (e.g. the node lost its job store): fail fast.
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if (status is not None and 400 <= status < 500) or (
                time.monotonic() >= poll_deadline
            ):
                raise RuntimeError(
                    f"Node {role!r} job {job_id} lost while polling: {exc}"
                ) from exc
            time.sleep(settings.node_poll_interval)
            continue

        status = body.get("status")
        if status == "done":
            return decode_update(body["result"] or {})
        if status == "error":
            raise RuntimeError(
                f"Node {role!r} job {job_id} failed: {body.get('error')} "
                f"(check: podman logs aidevteam_{role}_1)"
            )
        if time.monotonic() >= poll_deadline:
            raise RuntimeError(
                f"Node {role!r} job {job_id} did not finish within "
                f"{settings.node_max_wait:.0f}s "
                f"(check: podman logs aidevteam_{role}_1)"
            )
        time.sleep(settings.node_poll_interval)
