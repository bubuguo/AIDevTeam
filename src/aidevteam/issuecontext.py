"""GitHub issue context: fetch transient run state from GitHub.

The task, plan and feedback are transient context read from the issue
and never checkpointed.
"""

from dataclasses import dataclass, field
from typing import Any

from aidevteam.github_issue import get_github_issue_tool
from aidevteam.github_workflow import parse_task_body, parse_workflow_comment
from aidevteam.state import Plan


@dataclass
class IssueContext:
    """The transient context of a run, read from its GitHub issue.

    Nothing in here is checkpointed — it is rebuilt from the issue every
    time a node needs it.
    """

    issue_id: int
    task: str
    plan: Plan | None = None
    # The id of the latest plan comment; decisions are only honoured when
    # they were posted after it.
    plan_comment_id: int = 0
    # The feedback of every rejection so far, oldest first.
    plan_feedback: list[str] = field(default_factory=list)
    # Every decision comment, oldest first.
    decisions: list[dict[str, Any]] = field(default_factory=list)

    def decision_after(self, comment_id: int) -> dict[str, Any] | None:
        """The newest decision posted after ``comment_id``, if any."""
        found: dict[str, Any] | None = None
        for decision in self.decisions:
            if decision["comment_id"] > comment_id:
                found = decision
        return found


def fetch_issue_context(issue_id: int) -> IssueContext:
    """Read a run's transient context from its GitHub issue.

    The issue body carries the task; the comments carry the plan drafts and
    the human's decisions.

    Args:
        issue_id: The GitHub issue number, which is also the run's id.

    Returns:
        The context of the run.
    """
    with get_github_issue_tool() as tool:
        issue = tool.get_issue(issue_id)
        comments = tool.get_issue_comments(issue_id)

    context = IssueContext(
        issue_id=issue_id,
        task=parse_task_body(issue.body or ""),
    )
    for comment in comments:
        payload = parse_workflow_comment(comment.body)
        if not payload:
            continue
        kind = payload.get("kind")
        if kind == "plan":
            # The latest plan comment wins, even if its JSON block is broken:
            # the comment id still marks where a decision has to come from.
            context.plan_comment_id = comment.id
            plan_data = payload.get("plan")
            if plan_data:
                try:
                    context.plan = Plan.model_validate(plan_data)
                except Exception:  # noqa: BLE001 - a broken plan must not break the read
                    context.plan = None
        elif kind == "decision":
            action = str(payload.get("action") or "")
            feedback = str(payload.get("feedback") or "").strip()
            context.decisions.append(
                {
                    "comment_id": comment.id,
                    "action": action,
                    "feedback": feedback,
                }
            )
            if action == "reject" and feedback:
                context.plan_feedback.append(feedback)
    return context