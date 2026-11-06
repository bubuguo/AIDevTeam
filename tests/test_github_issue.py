"""Unit tests for the GitHub issue tool (HTTP layer mocked, no network)."""

import httpx

from aidevteam.github_issue import GitHubIssueTool


def _comment(comment_id: int, created_at: str) -> dict:
    return {
        "id": comment_id,
        "body": f"comment {comment_id}",
        "created_at": created_at,
        "updated_at": created_at,
        "html_url": f"https://github.com/o/r/issues/1#issuecomment-{comment_id}",
    }


def _tool_with_comment_pages(monkeypatch, pages: list[list[dict]]) -> GitHubIssueTool:
    """A tool whose HTTP layer serves the given comment pages, in order."""
    tool = GitHubIssueTool.__new__(GitHubIssueTool)  # skip __init__: no settings/client
    tool.token = "t"
    tool.owner = "o"
    tool.repo = "r"
    responses = iter(pages)
    calls: list[dict] = []

    def fake_request(method, url, *, allowed_statuses=(), **kwargs):
        calls.append(kwargs.get("params") or {})
        return httpx.Response(200, json=next(responses), request=httpx.Request(method, url))

    monkeypatch.setattr(tool, "_request", fake_request)
    tool.calls = calls  # type: ignore[attr-defined]
    return tool


def test_get_issue_comments_sorts_oldest_first(monkeypatch):
    """Replay safety (decision_after / plan_feedback) depends on oldest-first
    ordering — it is enforced here, not trusted from the API's default sort."""
    tool = _tool_with_comment_pages(
        monkeypatch,
        [
            [
                _comment(3, "2025-01-03T00:00:00Z"),
                _comment(1, "2025-01-01T00:00:00Z"),
                _comment(2, "2025-01-02T00:00:00Z"),
            ]
        ],
    )
    comments = tool.get_issue_comments(1)
    assert [comment.id for comment in comments] == [1, 2, 3]


def test_get_issue_comments_follows_pagination(monkeypatch):
    tool = _tool_with_comment_pages(
        monkeypatch,
        [
            [_comment(1, "2025-01-01T00:00:00Z")],
            [_comment(2, "2025-01-02T00:00:00Z")],
            [],  # an empty page ends the walk
        ],
    )
    comments = tool.get_issue_comments(1, per_page=1)
    assert [comment.id for comment in comments] == [1, 2]
    assert [call.get("page") for call in tool.calls] == [1, 2, 3]