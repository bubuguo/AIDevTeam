"""Reviewer agent: judges the submission and either approves it or requests fixes."""

from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console

from aidevteam.llm import get_structured_llm
from aidevteam.state import Review, TeamState

console = Console()

REVIEWER_SYSTEM = """You are the senior Reviewer on a software team that builds small Python CLI applications.

Judge the implementation against the plan and the test results.
Approve only when:
- the code is complete and matches the plan,
- all tests passed and the CLI entry point runs,
- the code is reasonably clean (docstrings, no placeholders, no dead code).

If not approved, return a short, concrete, actionable list of fixes.
Do not demand gold-plating; this is a small CLI tool.

Note: file contents may contain a "[display truncated ...]" marker. That is a
display-length limit only — the files on disk are complete. Never request a
resubmission because of that marker; judge only the code you can see.
"""


MAX_REVIEW_CHARS_PER_FILE = 20000


def _render_for_review(path: str, content: str) -> str:
    """Render a file for the reviewer, marking display-only truncation clearly.

    The reviewer must never mistake a display limit for the file itself being
    incomplete, so any cut is annotated with an explicit marker.
    """
    if len(content) <= MAX_REVIEW_CHARS_PER_FILE:
        return f"### {path}\n{content}"
    return (
        f"### {path}\n{content[:MAX_REVIEW_CHARS_PER_FILE]}\n"
        f"... [display truncated: showing first {MAX_REVIEW_CHARS_PER_FILE} of "
        f"{len(content)} chars — the file itself is complete on disk]"
    )


def reviewer(state: TeamState) -> dict:
    console.print("[bold magenta]▶ Reviewer[/bold magenta] evaluating the submission…")
    plan = state["plan"]
    code = state["code"]
    report = state["test_report"]

    code_text = "\n\n".join(
        _render_for_review(f.path, f.content) for f in code.files
    )
    review: Review = get_structured_llm(Review).invoke(
        [
            SystemMessage(content=REVIEWER_SYSTEM),
            HumanMessage(
                content=(
                    f"Task:\n{state['task']}\n\n"
                    f"Plan ({plan.app_name}): {plan.summary}\n\n"
                    f"Implementation:\n{code_text}\n\n"
                    f"Test report: {report.summary}\n"
                    f"Test output (truncated):\n{report.output}"
                )
            ),
        ]
    )

    if review.approved:
        console.print("  [bold green]✔ approved[/bold green]")
    else:
        console.print("  [bold red]✘ revisions requested:[/bold red]")
        for item in review.feedback:
            console.print(f"    [red]-[/red] {item}")

    return {"review": review, "approved": review.approved, "feedback": review.feedback}
