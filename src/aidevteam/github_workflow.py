"""The versioned GitHub workflow format.

Everything the team writes to (or reads from) a GitHub issue follows one
versioned format, so a comment can be parsed back into structured data and
an older format can be rejected instead of silently misread.

Three kinds of payload exist:

**Task body** — the issue body created by ``run``::

    <!-- aidevteam: v1:task -->
    # Task

    build a word-count CLI tool

**Plan comment** — a plan draft or the final approved plan::

    <!-- aidevteam: v1:plan -->
    ## Draft Plan

    **App:** word-count
    ...

**Decision comment** — a human approving or rejecting a plan, written as a
slash command. The versioned marker form is recognized too, so a
machine-written decision parses the same way::

    /approve
    /reject please add an edge case for empty input

The version marker is an HTML comment so it stays invisible when GitHub
renders the issue, while remaining greppable in the raw API response.
"""

import json
import re
from typing import Any

# The current workflow format version. Bump this when the format changes in
# a way that older readers cannot understand.
WORKFLOW_VERSION = "v1"

# Marker prefix shared by every machine-written payload.
MARKER_PREFIX = "aidevteam"

# ``<!-- aidevteam: v1:kind[:payload] -->``
_MARKER_RE = re.compile(
    r"<!--\s*" + MARKER_PREFIX + r":\s*(?P<version>\w+):(?P<kind>\w+)(?::(?P<payload>[^\n]*?))?\s*-->"
)

# A human decision: ``/approve`` or ``/reject <feedback>``.
_DECISION_RE = re.compile(
    r"^\s*/(?P<action>approve|reject)\b(?P<feedback>.*)$",
    re.IGNORECASE | re.DOTALL,
)

# The task body is everything after the marker, minus the ``# Task`` heading.
_TASK_HEADING_RE = re.compile(r"^\s*#\s*Task\s*$", re.IGNORECASE | re.MULTILINE)

# The machine-readable plan embedded in a plan comment.
_PLAN_JSON_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def marker(kind: str, payload: str = "") -> str:
    """Build a versioned workflow marker.

    Args:
        kind: The payload kind (``task``, ``plan``, ``decision``).
        payload: Optional inline payload (e.g. the decision action).

    Returns:
        The marker string, e.g. ``<!-- aidevteam: v1:decision:approve -->``.
    """
    suffix = f":{payload}" if payload else ""
    return f"<!-- {MARKER_PREFIX}: {WORKFLOW_VERSION}:{kind}{suffix} -->"


def format_task_body(task: str) -> str:
    """Format a task as a versioned GitHub issue body.

    Args:
        task: The natural-language task description.

    Returns:
        The issue body, carrying the task marker.
    """
    return f"{marker('task')}\n\n# Task\n\n{task.strip()}\n"


def format_plan_comment(plan: dict[str, Any], stage: str = "Draft Plan") -> str:
    """Format a plan as a versioned GitHub comment.

    The comment carries two representations of the plan: a readable Markdown
    summary for the human reviewer, and a fenced JSON block for the machine.
    The JSON block is what lets a later ``resume`` rebuild the plan — the
    plan itself is never checkpointed.

    Args:
        plan: The plan as a plain dict.
        stage: The heading for this plan (``Draft Plan``, ``Final Plan``).

    Returns:
        The comment body: Markdown, then the JSON block, then the marker.
    """
    lines = [f"## {stage}", ""]

    app_name = plan.get("app_name")
    if app_name:
        lines.append(f"**App:** {app_name}")
        lines.append("")

    summary = plan.get("summary")
    if summary:
        lines.append(str(summary))
        lines.append("")

    commands = plan.get("cli_commands") or []
    if commands:
        lines.append("**Example commands:**")
        lines.extend(f"- `{command}`" for command in commands)
        lines.append("")

    files = plan.get("files") or []
    if files:
        lines.append("**Files:**")
        for spec in files:
            path = spec.get("path") if isinstance(spec, dict) else getattr(spec, "path", "")
            description = (
                spec.get("description")
                if isinstance(spec, dict)
                else getattr(spec, "description", "")
            )
            lines.append(f"- `{path}` — {description}")
        lines.append("")

    lines.append(
        "Reply with `/approve` to start coding, or `/reject <reason>` to "
        "send the plan back to the Planner."
    )
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(plan, indent=2, default=str, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append(marker("plan"))
    return "\n".join(lines)


def parse_workflow_comment(body: str) -> dict[str, Any] | None:
    """Parse a GitHub comment into a workflow payload.

    Recognizes both the machine-written versioned markers and the human
    slash commands. A marker whose version is not the current one is
    rejected (returns None) rather than misread.

    Args:
        body: The raw comment body.

    Returns:
        A payload dict, or None when the comment is not a workflow comment
        or carries an unsupported version.

        * ``{"kind": "decision", "action": "approve"|"reject",
           "feedback": str, "version": str}``
        * ``{"kind": "plan", "version": str}``
        * ``{"kind": "task", "task": str, "version": str}``
    """
    if not body:
        return None

    match = _MARKER_RE.search(body)
    if match:
        version = match.group("version")
        if version != WORKFLOW_VERSION:
            # An unknown format version must not be guessed at.
            return None
        kind = match.group("kind")
        if kind == "decision":
            action = (match.group("payload") or "").strip().lower()
            if action not in ("approve", "reject"):
                return None
            return {
                "kind": "decision",
                "action": action,
                "feedback": _decision_feedback(body),
                "version": version,
            }
        if kind == "plan":
            return {
                "kind": "plan",
                "plan": parse_plan_json(body),
                "version": version,
            }
        if kind == "task":
            return {
                "kind": "task",
                "task": parse_task_body(body),
                "version": version,
            }
        return None

    # A human decision written as a slash command.
    decision = _DECISION_RE.match(body)
    if decision:
        action = decision.group("action").lower()
        feedback = decision.group("feedback").strip()
        return {
            "kind": "decision",
            "action": action,
            "feedback": feedback,
            "version": WORKFLOW_VERSION,
        }
    return None


def parse_plan_json(body: str) -> dict[str, Any] | None:
    """Extract the machine-readable plan from a plan comment.

    The plan comment embeds the plan as a fenced ``json`` block; that block
    is the durable copy of the plan, since the plan is never checkpointed.

    Args:
        body: The raw comment body.

    Returns:
        The plan as a dict, or None when the block is missing or malformed.
    """
    match = _PLAN_JSON_RE.search(body or "")
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_task_body(body: str) -> str:
    """Extract the task text from a versioned issue body.

    Args:
        body: The raw issue body.

    Returns:
        The task text, or an empty string when the body carries no task.
    """
    if not body:
        return ""
    match = _MARKER_RE.search(body)
    text = body
    if match:
        text = body[match.end():]
    # Drop the ``# Task`` heading the formatter adds.
    text = _TASK_HEADING_RE.sub("", text, count=1)
    return text.strip()


def _decision_feedback(body: str) -> str:
    """The feedback text of a decision comment, without the marker."""
    text = body
    match = _MARKER_RE.search(body)
    if match:
        text = body[match.end():]
    decision = _DECISION_RE.match(text.strip())
    if decision:
        return decision.group("feedback").strip()
    return ""


def plan_to_dict(plan: Any) -> dict[str, Any]:
    """Normalize a plan (model instance or dict) into a plain dict."""
    if isinstance(plan, dict):
        return plan
    if hasattr(plan, "model_dump"):
        return plan.model_dump()
    return json.loads(json.dumps(plan, default=str))
