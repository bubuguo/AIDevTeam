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
from aidevteam.state import TeamState

# Model-typed TeamState fields, derived from the state definition so new
# model fields work over RPC without touching this module.
_STATE_MODELS: dict[str, type[BaseModel]] = {
    name: ann
    for name, ann in get_type_hints(TeamState).items()
    if isinstance(ann, type) and issubclass(ann, BaseModel)
}


def encode_state(state: dict[str, Any]) -> dict[str, Any]:
    """Convert a state dict (pydantic model instances included) to JSON-safe values."""
    encoded: dict[str, Any] = {}
    for key, value in state.items():
        model = _STATE_MODELS.get(key)
        if model is not None and isinstance(value, model):
            encoded[key] = value.model_dump()
        else:
            encoded[key] = value
    return encoded


def decode_update(update: dict[str, Any]) -> dict[str, Any]:
    """Convert a JSON state update back into a dict with pydantic model instances."""
    decoded: dict[str, Any] = {}
    for key, value in update.items():
        model = _STATE_MODELS.get(key)
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
    while True:
        try:
            response = httpx.post(f"{base}/run", json=payload, timeout=30.0)
            response.raise_for_status()
            job_id = response.json()["job_id"]
            break
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
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
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            # Transient blip while polling — keep waiting until the deadline.
            if time.monotonic() >= poll_deadline:
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
