"""Tester agent: writes pytest tests, executes them, and reports real results."""

import sys

from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console

from aidevteam.llm import get_structured_llm
from aidevteam.state import TeamState, TestReport, TestSuite
from aidevteam.workspace import run_command, workspace_for, write_files

console = Console()

TESTER_SYSTEM = """You are the Tester on a software team that builds small Python CLI applications.

Write a focused pytest suite for the implemented code.
Rules:
- Test files must be named `test_*.py` and live flat in the workspace root.
- Import the application modules directly (e.g. `from app import convert`).
- Cover the happy path plus at least one edge case per major function.
- You may test CLI behaviour via subprocess: `subprocess.run([sys.executable, "app.py", ...])`.
- Only use pytest and the Python standard library.
- Return the FULL content of every test file.
"""


def tester(state: TeamState) -> dict:
    console.print("[bold yellow]▶ Tester[/bold yellow] writing and running tests…")
    workspace = workspace_for(state["run_id"])
    plan = state["plan"]
    code = state["code"]

    code_text = "\n\n".join(f"### {f.path}\n{f.content}" for f in code.files)
    suite: TestSuite = get_structured_llm(TestSuite).invoke(
        [
            SystemMessage(content=TESTER_SYSTEM),
            HumanMessage(
                content=(
                    f"Task:\n{state['task']}\n\n"
                    f"Plan ({plan.app_name}): {plan.summary}\n"
                    f"Example commands: {', '.join(plan.cli_commands) or 'n/a'}\n\n"
                    f"Implementation to test:\n{code_text}"
                )
            ),
        ]
    )
    # Remove stale test files from previous iterations so pytest only
    # collects the current suite (old tests may target outdated APIs).
    for stale in workspace.glob("test_*.py"):
        stale.unlink()
        console.print(f"  [dim]removed stale[/dim] {stale.name}")

    write_files(workspace, suite.files)
    for file in suite.files:
        console.print(f"  [yellow]wrote[/yellow] {file.path} ({len(file.content)} chars)")

    # Run the real test suite.
    exit_code, output = run_command(
        [sys.executable, "-m", "pytest", "-q", "--tb=short"], workspace
    )
    passed = exit_code == 0

    # Smoke-test the CLI entry point as well.
    smoke = ""
    if (workspace / "app.py").exists():
        smoke_code, smoke_out = run_command([sys.executable, "app.py", "--help"], workspace)
        smoke = f"\n--- smoke test: app.py --help (exit {smoke_code}) ---\n{smoke_out}"
        if smoke_code != 0:
            passed = False

    summary = (
        f"pytest exit {exit_code}: all tests passed"
        if passed
        else f"pytest exit {exit_code}: tests failed"
    )
    mark = "[green]✔[/green]" if passed else "[red]✘[/red]"
    console.print(f"  {mark} {summary}")

    report = TestReport(passed=passed, summary=summary, output=(output + smoke)[-4000:])
    return {"test_report": report}
