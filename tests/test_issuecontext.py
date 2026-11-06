"""Unit tests for the transient issue context read from GitHub."""

from aidevteam.github_issue import Comment, Issue
from aidevteam.github_workflow import format_plan_comment, format_task_body
from aidevteam.issuecontext import fetch_issue_context


def _issue() -> Issue:
    return Issue(
        number=12345,
        title="build a demo",
        body=format_task_body("build a demo"),
        state="open",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        html_url="https://github.com/o/r/issues/12345",
    )


def _comment(comment_id: int, body: str) -> Comment:
    return Comment(
        id=comment_id,
        body=body,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        html_url=f"https://github.com/o/r/issues/12345#issuecomment-{comment_id}",
    )


def _plan_comment(comment_id: int, with_json: bool = True) -> Comment:
    if with_json:
        body = format_plan_comment(
            {"app_name": "demo", "summary": "demo app", "files": []},
            stage="Draft Plan",
        )
    else:
        # A plan draft whose machine-readable JSON block is missing.
        body = "## Draft Plan\n\n(no machine-readable block)\n\n<!-- aidevteam: v1:plan -->"
    return _comment(comment_id, body)


class _FakeTool:
    """Stand-in for GitHubIssueTool serving a canned issue + comments."""

    def __init__(self, comments: list[Comment]) -> None:
        self._comments = comments

    def get_issue(self, issue_number: int) -> Issue:
        return _issue()

    def get_issue_comments(self, issue_number: int) -> list[Comment]:
        return list(self._comments)


def _fetch(comments: list[Comment]):
    return fetch_issue_context(12345, tool=_FakeTool(comments))


def test_latest_plan_comment_wins():
    context = _fetch([_plan_comment(1), _plan_comment(2)])
    assert context.plan is not None
    assert context.plan.app_name == "demo"
    assert context.plan_comment_id == 2


def test_plan_comment_without_json_drops_the_previous_plan():
    """A newer plan comment replaces the old plan even when its own JSON
    block is missing — otherwise the coder would implement a stale plan
    while the human approves the unreadable new draft."""
    context = _fetch([_plan_comment(1), _plan_comment(2, with_json=False)])
    assert context.plan is None
    assert context.plan_comment_id == 2


def test_rejection_feedback_accumulates_oldest_first():
    context = _fetch(
        [
            _plan_comment(1),
            _comment(2, "/reject too many files"),
            _comment(3, "/reject smaller scope"),
        ]
    )
    assert context.plan_feedback == ["too many files", "smaller scope"]


def test_feedback_addressed_by_a_newer_plan_is_dropped():
    """Feedback the latest plan draft was written to address must not be
    re-fed to the planner on the next revision."""
    context = _fetch(
        [
            _plan_comment(1),
            _comment(2, "/reject too many files"),
            _plan_comment(3),
            _comment(4, "/reject smaller scope"),
        ]
    )
    assert context.plan_feedback == ["smaller scope"]


def test_decision_after_ignores_decisions_on_older_drafts():
    """Only decisions posted after the latest plan comment are honoured."""
    context = _fetch([_comment(1, "/approve"), _plan_comment(2)])
    assert context.decision_after(context.plan_comment_id) is None
    assert context.decision_after(0)["action"] == "approve"