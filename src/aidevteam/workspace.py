"""Sandbox workspace helpers: safe file writing and subprocess execution."""

import subprocess
import sys
import time
from pathlib import Path

from aidevteam.state import GeneratedFile

MAX_OUTPUT_CHARS = 6000


def create_workspace(root: Path | None = None) -> Path:
    """Create a timestamped sandbox directory under ./generated/."""
    base = root or Path.cwd() / "generated"
    workspace = base / time.strftime("%Y%m%d-%H%M%S")
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


def python_executable() -> str:
    return sys.executable
