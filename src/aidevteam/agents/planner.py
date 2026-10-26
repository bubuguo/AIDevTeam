"""Planner agent: turns a natural-language task into a structured build plan."""

from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console

from aidevteam.llm import get_structured_llm
from aidevteam.state import Plan, TeamState

console = Console()

PLANNER_SYSTEM = """You are the Planner on a software team that builds small Python CLI applications.

Given a task, produce a concise, buildable plan:
- Keep the project small: 1-3 flat Python files in the workspace root.
- The CLI entry point MUST be `app.py`, built with the standard library `argparse`.
- Generated apps must run with the Python standard library only (no third-party imports).
- `app.py` must expose reusable functions (pure logic kept separate from argument
  parsing) so an automated test suite can import them directly.
- List every file with its path and responsibility, and give example CLI invocations.
"""


def planner(state: TeamState) -> dict:
    console.print("[bold cyan]▶ Planner[/bold cyan] drafting the build plan…")
    llm = get_structured_llm(Plan)
    plan: Plan = llm.invoke(
        [
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(content=f"Task:\n{state['task']}"),
        ]
    )
    console.print(f"  [cyan]app:[/cyan] {plan.app_name} — {plan.summary}")
    for spec in plan.files:
        console.print(f"  [cyan]file:[/cyan] {spec.path} — {spec.description}")
    return {"plan": plan}
