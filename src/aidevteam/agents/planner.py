"""Planner agent: turns a natural-language task into a structured build plan."""

import json

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from rich.console import Console

from aidevteam.github_issue import get_github_issue_tool
from aidevteam.github_workflow import format_plan_comment, plan_to_dict
from aidevteam.issuecontext import fetch_issue_context
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
    issue_id = state["issue_id"]
    context = fetch_issue_context(issue_id)

    feedback = context.plan_feedback or []
    previous = context.plan
    task_text = f"Task:\n{context.task}"
    if feedback:
        # Previous plan rejected at the gate — revision must address all feedback.
        console.print(
            "[bold cyan]  ↻ previous plan rejected — addressing feedback:[/bold cyan]"
        )
        for item in feedback:
            console.print(f"    [yellow]-[/yellow] {item}")
        previous_json = (
            previous.model_dump_json(indent=2)
            if isinstance(previous, BaseModel)
            else json.dumps(previous, indent=2, default=str)
        )
        task_text += (
            "\n\nYour previous plan was rejected by the team. Here it is:\n\n"
            + previous_json
            + "\n\nProduce a new plan that keeps what works and addresses "
            + "ALL of this feedback:\n"
            + "\n".join(f"- {item}" for item in feedback)
        )
    llm = get_structured_llm(Plan)
    plan: Plan = llm.invoke(
        [
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(content=task_text),
        ]
    )
    console.print(f"  [cyan]app:[/cyan] {plan.app_name} — {plan.summary}")
    for spec in plan.files:
        console.print(f"  [cyan]file:[/cyan] {spec.path} — {spec.description}")
    
    # Post the plan to the GitHub issue
    body = format_plan_comment(plan_to_dict(plan), stage="Draft Plan")
    with get_github_issue_tool() as tool:
        tool.add_comment(issue_id, body)
    
    # The plan is transient context: the graph posts it to the GitHub issue
    # and never writes it to the checkpoint.
    return {}
