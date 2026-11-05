"""Sandbox workspace helpers: safe file writing and subprocess execution."""

import os
import subprocess
from pathlib import Path

from aidevteam.state import GeneratedFile

MAX_OUTPUT_CHARS = 6000


def workspace_root(root: Path | None = None) -> Path:
    """The workspace base directory.

    Defaults to ``./generated`` and can be overridden with the
    AIDEVTEAM_WORKSPACE_ROOT env var (containers mount the shared workspace
    volume at /workspace).
    """
    return root or Path(os.environ.get("AIDEVTEAM_WORKSPACE_ROOT") or Path.cwd() / "generated")


def workspace_for(issue_id: int, root: Path | None = None) -> Path:
    """The workspace directory for an issue: ``<root>/<issue_id>``.

    The GitHub issue number is the only identifier, so every node derives the
    same workspace path from it.
    """
    return workspace_root(root) / str(issue_id)


def create_workspace(issue_id: int, root: Path | None = None) -> Path:
    """Create (idempotently) the workspace directory for an issue.

    Idempotent because the Coder re-derives the workspace on every revision
    iteration.
    """
    workspace = workspace_for(issue_id, root)
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _safe_path(workspace: Path, relative: str) -> Path:
    """Resolve `relative` inside `workspace`, refusing escapes."""
    candidate = (workspace / relative).resolve()
    resolved_ws = workspace.resolve()
    if candidate != resolved_ws and resolved_ws not in candidate.parents:
        raise ValueError(f"Refusing to write outside the workspace: {relative!r}")
    return candidate


def write_files(workspace: Path, files: list[GeneratedFile]) -> list[Path]:
    """Write generated files into the workspace; returns the written paths."""
    written = []
    for file in files:
        target = _safe_path(workspace, file.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(file.content, encoding="utf-8")
        written.append(target)
    return written


def run_command(cmd: list[str], cwd: Path, timeout: int = 180) -> tuple[int, str]:
    """Run a command in the workspace; returns (exit_code, truncated combined output)."""
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
        )
        output = (proc.stdout + "\n" + proc.stderr).strip()
        return proc.returncode, output[-MAX_OUTPUT_CHARS:]
    except subprocess.TimeoutExpired:
        return 124, f"Command timed out after {timeout}s: {' '.join(cmd)}"
    except OSError as exc:
        return 1, f"Failed to run command: {exc}"
