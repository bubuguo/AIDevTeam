"""Coder agent: implements the planned files, optionally applying reviewer feedback."""

from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console

from aidevteam.config import get_settings
from aidevteam.llm import get_structured_llm
from aidevteam.state import CodeResult, TeamState
from aidevteam.workspace import create_workspace, write_files

console = Console()

CODER_SYSTEM = """You are the Coder on a software team that builds small Python CLI applications.

Implement every file from the plan as complete, runnable code.
Rules:
- All files live flat in the workspace root; the CLI entry point is `app.py` using `argparse`.
- Standard library only — no third-party imports in generated code.
- Separate pure logic from argument parsing so tests can import functions directly.
- Include module docstrings and keep functions small and focused.
- If revision feedback is provided, fix exactly those issues while keeping the rest working.
- Return the FULL content of every file (no placeholders, no "..." omissions).
"""


def coder(state: TeamState) -> dict:
    settings = get_settings()
    iteration = state.get("iteration", 0) + 1
    feedback = state.get("feedback") or []

    if iteration == 1:
        console.print("[bold green]▶ Coder[/bold green] implementing the plan…")
    else:
        console.print(
            f"[bold green]▶ Coder[/bold green] revision {iteration}/{settings.max_iterations}…"
        )

    workspace = create_workspace(state["run_id"])

    plan = state["plan"]
    plan_text = "\n".join(f"- {f.path}: {f.description}" for f in plan.files)
    messages = [
        SystemMessage(content=CODER_SYSTEM),
        HumanMessage(
            content=(
                f"Task:\n{state['task']}\n\n"
                f"Plan ({plan.app_name}): {plan.summary}\n"
                f"Example commands: {', '.join(plan.cli_commands) or 'n/a'}\n"
                f"Files to implement:\n{plan_text}"
            )
        ),
    ]
    if feedback:
        messages.append(
            HumanMessage(
                content="Reviewer feedback to address:\n- " + "\n- ".join(feedback)
            )
        )

    result: CodeResult = get_structured_llm(CodeResult).invoke(messages)
    written = write_files(workspace, result.files)
    for path in written:
        console.print(f"  [green]wrote[/green] {path.name} ({path.stat().st_size} bytes)")
    if result.notes:
        console.print(f"  [dim]notes: {result.notes}[/dim]")

    return {
        "code": result,
        "iteration": iteration,
        "feedback": [],  # consumed
    }
