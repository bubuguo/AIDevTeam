"""GitHub Issue Tool: manage issues via GitHub REST API.

Provides functionality to create/close issues, add comments and labels,
and retrieve issue details and comments.

The versioned workflow format used for task bodies, plan comments, and
``/approve`` / ``/reject`` decisions lives in ``github_workflow.py``.
"""

from typing import Any, Self, TypeVar

import httpx
from pydantic import BaseModel, Field, ValidationError

from aidevteam.config import get_settings
from aidevteam.github_workflow import parse_workflow_comment


class GitHubIssueError(RuntimeError):
    """A GitHub issue operation failed."""


ModelT = TypeVar("ModelT", bound=BaseModel)


class Issue(BaseModel):
    """Represents a GitHub issue."""

    number: int = Field(description="Issue number")
    title: str = Field(description="Issue title")
    body: str | None = Field(default=None, description="Issue body/description")
    state: str = Field(description="Issue state: 'open' or 'closed'")
    labels: list[dict[str, Any]] = Field(default_factory=list, description="Issue labels")
    comments: int = Field(default=0, description="Number of comments")
    created_at: str = Field(description="Issue creation timestamp")
    updated_at: str = Field(description="Issue update timestamp")
    user: dict[str, Any] = Field(default_factory=dict, description="Issue creator")
    html_url: str = Field(description="URL to view the issue in a browser")


class Comment(BaseModel):
    """Represents a GitHub issue comment."""

    id: int = Field(description="Comment ID")
    body: str = Field(description="Comment body")
    user: dict[str, Any] = Field(default_factory=dict, description="Comment author")
    created_at: str = Field(description="Comment creation timestamp")
    updated_at: str = Field(description="Comment update timestamp")
    html_url: str = Field(description="URL to view the comment")


class GitHubIssueTool:
    """Tool for managing GitHub issues via the REST API.

    Provides methods to create/close issues, add comments and labels,
    and retrieve issue details and comments.

    Usage:
        tool = GitHubIssueTool()
        issue = tool.create_issue("Bug title", "Bug description")
        tool.add_comment(issue.number, "Additional info")
        tool.add_labels(issue.number, ["bug", "priority:high"])
    """

    BASE_URL = "https://api.github.com"
    DEFAULT_TIMEOUT = 30.0

    def __init__(self, token: str | None = None, owner: str | None = None, repo: str | None = None):
        """Initialize the GitHub Issue Tool.

        Args:
            token: GitHub personal access token. If not provided, uses GITHUB_TOKEN from settings.
            owner: Repository owner. If not provided, uses GITHUB_OWNER from settings.
            repo: Repository name. If not provided, uses GITHUB_REPO from settings.
        """
        settings = get_settings()
        self.token = token or settings.github_token
        self.owner = owner or settings.github_owner
        self.repo = repo or settings.github_repo
        self._client = httpx.Client(
            base_url=self.BASE_URL,
            headers=self._build_headers(),
            timeout=self.DEFAULT_TIMEOUT,
        )

    def _build_headers(self) -> dict[str, str]:
        """Build request headers with authentication."""
        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _check_config(self) -> None:
        """Validate that required configuration is present."""
        if not self.token:
            raise GitHubIssueError(
                "GitHub token is required. Set GITHUB_TOKEN in environment or .env file."
            )
        if not self.owner:
            raise GitHubIssueError(
                "GitHub owner is required. Set GITHUB_OWNER in environment or .env file."
            )
        if not self.repo:
            raise GitHubIssueError(
                "GitHub repo is required. Set GITHUB_REPO in environment or .env file."
            )

    def _request(
        self,
        method: str,
        url: str,
        *,
        allowed_statuses: tuple[int, ...] = (),
        **kwargs: Any,
    ) -> httpx.Response:
        """Execute a GitHub request and translate transport failures."""
        try:
            response = getattr(self._client, method)(url, **kwargs)
            if response.status_code not in allowed_statuses:
                response.raise_for_status()
            return response
        except httpx.HTTPError as exc:
            raise GitHubIssueError(f"GitHub request failed: {exc}") from exc

    def _parse_model(
        self,
        response: httpx.Response,
        model_type: type[ModelT],
        operation: str,
    ) -> ModelT:
        """Parse and validate a GitHub response model."""
        try:
            return model_type.model_validate(response.json())
        except (ValidationError, ValueError, TypeError) as exc:
            raise GitHubIssueError(
                f"Invalid GitHub response for {operation}: {exc}"
            ) from exc

    def _parse_models(
        self,
        response: httpx.Response,
        model_type: type[ModelT],
        operation: str,
    ) -> list[ModelT]:
        """Parse and validate a list of GitHub response models."""
        try:
            data = response.json()
            if not isinstance(data, list):
                raise TypeError("expected a JSON list")
            return [model_type.model_validate(item) for item in data]
        except (ValidationError, ValueError, TypeError) as exc:
            raise GitHubIssueError(
                f"Invalid GitHub response for {operation}: {exc}"
            ) from exc

    def _get_issue_url(self, issue_number: int) -> str:
        """Build the URL for a specific issue."""
        return f"/repos/{self.owner}/{self.repo}/issues/{issue_number}"

    def create_issue(self, title: str, body: str | None = None, labels: list[str] | None = None) -> Issue:
        """Create a new GitHub issue.

        Args:
            title: Issue title.
            body: Issue description/body (optional).
            labels: List of label names to apply (optional).

        Returns:
            The created Issue object.
        """
        self._check_config()

        payload: dict[str, Any] = {"title": title}
        if body:
            payload["body"] = body
        if labels:
            payload["labels"] = labels

        response = self._request(
            "post",
            f"/repos/{self.owner}/{self.repo}/issues",
            json=payload,
        )
        return self._parse_model(response, Issue, "issue creation")

    def create_sub_issue(
        self,
        title: str,
        parent_issue_number: int,
        body: str | None = None,
        labels: list[str] | None = None,
    ) -> Issue:
        """Create a sub-issue linked to a parent issue.

        The sub-issue is linked to the parent via a reference in the body.

        Args:
            title: Issue title.
            parent_issue_number: Number of the parent issue.
            body: Additional issue description (optional).
            labels: List of label names to apply (optional).

        Returns:
            The created Issue object.
        """
        self._check_config()

        # Build body with parent reference
        parent_ref = f"Related to #{parent_issue_number}"
        if body:
            full_body = f"{parent_ref}\n\n{body}"
        else:
            full_body = parent_ref

        # Add "sub-issue" label by default
        all_labels = (labels or []) + ["sub-issue"]

        return self.create_issue(title, full_body, all_labels)

    def close_issue(self, issue_number: int) -> Issue:
        """Close an existing issue.

        Args:
            issue_number: Number of the issue to close.

        Returns:
            The updated Issue object (now closed).
        """
        self._check_config()

        response = self._request(
            "patch",
            self._get_issue_url(issue_number),
            json={"state": "closed"},
        )
        return self._parse_model(response, Issue, "issue closing")

    def reopen_issue(self, issue_number: int) -> Issue:
        """Reopen a closed issue.

        Args:
            issue_number: Number of the issue to reopen.

        Returns:
            The updated Issue object (now open).
        """
        self._check_config()

        response = self._request(
            "patch",
            self._get_issue_url(issue_number),
            json={"state": "open"},
        )
        return self._parse_model(response, Issue, "issue reopening")

    def add_comment(self, issue_number: int, body: str) -> Comment:
        """Add a comment to an issue.

        Args:
            issue_number: Number of the issue to comment on.
            body: Comment text.

        Returns:
            The created Comment object.
        """
        self._check_config()

        response = self._request(
            "post",
            f"/repos/{self.owner}/{self.repo}/issues/{issue_number}/comments",
            json={"body": body},
        )
        return self._parse_model(response, Comment, "issue comment")

    def add_labels(self, issue_number: int, labels: list[str]) -> Issue:
        """Add labels to an issue.

        Args:
            issue_number: Number of the issue.
            labels: List of label names to add.

        Returns:
            The updated Issue object.
        """
        self._check_config()

        response = self._request(
            "post",
            self._get_issue_url(issue_number),
            json={"labels": labels},
        )
        return self._parse_model(response, Issue, "label addition")

    def remove_labels(self, issue_number: int, labels: list[str]) -> Issue:
        """Remove labels from an issue.

        Args:
            issue_number: Number of the issue.
            labels: List of label names to remove.

        Returns:
            The updated Issue object.
        """
        self._check_config()

        # GitHub API requires removing labels one at a time via DELETE
        for label in labels:
            self._request(
                "delete",
                f"/repos/{self.owner}/{self.repo}/issues/{issue_number}/labels/{label}",
                allowed_statuses=(200, 404),
            )

        # Get updated issue
        return self.get_issue(issue_number)

    def get_issue(self, issue_number: int) -> Issue:
        """Get details of a specific issue.

        Args:
            issue_number: Number of the issue.

        Returns:
            The Issue object.
        """
        self._check_config()

        response = self._request("get", self._get_issue_url(issue_number))
        return self._parse_model(response, Issue, "issue retrieval")

    def get_issue_comments(
        self,
        issue_number: int,
        per_page: int = 100,
        max_pages: int = 20,
    ) -> list[Comment]:
        """Get all comments on an issue, following pagination.

        The GitHub comments endpoint returns at most ``per_page`` comments per
        request and signals more pages through the ``Link`` header. Without
        pagination a busy issue silently loses its later comments — including
        the approval decision the workflow is waiting for.

        Args:
            issue_number: Number of the issue.
            per_page: Comments requested per API call (GitHub caps this at 100).
            max_pages: Safety bound on the number of pages fetched.

        Returns:
            List of Comment objects, sorted by creation date (oldest first).
        """
        self._check_config()

        comments: list[Comment] = []
        page = 1
        while page <= max_pages:
            response = self._request(
                "get",
                f"/repos/{self.owner}/{self.repo}/issues/{issue_number}/comments",
                params={"per_page": per_page, "page": page},
            )
            batch = self._parse_models(response, Comment, "issue comment retrieval")
            comments.extend(batch)
            if len(batch) < per_page:
                break
            page += 1
        return comments

    def get_workflow_decision(
        self,
        issue_number: int,
        after_comment_id: int = 0,
    ) -> dict[str, Any] | None:
        """Find the newest ``/approve`` or ``/reject`` decision on an issue.

        Only comments created after ``after_comment_id`` are considered, so a
        decision that was already consumed is never replayed. The newest
        matching decision wins.

        Args:
            issue_number: Number of the issue.
            after_comment_id: Ignore comments with an ID at or below this value.

        Returns:
            ``{"action": "approve"|"reject", "feedback": str, "comment_id": int}``
            or None when no decision was found.
        """
        decision: dict[str, Any] | None = None
        for comment in self.get_issue_comments(issue_number):
            if comment.id <= after_comment_id:
                continue
            parsed = parse_workflow_comment(comment.body)
            if not parsed or parsed.get("kind") != "decision":
                continue
            action = str(parsed.get("action") or "").strip().lower()
            if action not in ("approve", "reject"):
                continue
            decision = {
                "action": action,
                "feedback": str(parsed.get("feedback") or ""),
                "comment_id": comment.id,
            }
        return decision

    def get_sub_issues(self, label: str = "sub-issue") -> list[Issue]:
        """Get all sub-issues in the repository.

        Sub-issues are identified by a specific label (default: "sub-issue").

        Args:
            label: Label used to identify sub-issues.

        Returns:
            List of Issue objects that are sub-issues.
        """
        self._check_config()

        response = self._request(
            "get",
            f"/repos/{self.owner}/{self.repo}/issues",
            params={"labels": label, "state": "open"},
        )
        return self._parse_models(response, Issue, "sub-issue retrieval")

    def list_issues(
        self,
        state: str = "open",
        labels: list[str] | None = None,
        creator: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[Issue]:
        """List issues in the repository.

        Args:
            state: Issue state filter ('open', 'closed', or 'all').
            labels: Filter by labels (issues must have all specified labels).
            creator: Filter by issue creator username.
            since: ISO 8601 timestamp to filter issues updated after this time.
            until: ISO 8601 timestamp to filter issues updated before this time.

        Returns:
            List of Issue objects.
        """
        self._check_config()

        params: dict[str, Any] = {"state": state}
        if labels:
            params["labels"] = ",".join(labels)
        if creator:
            params["creator"] = creator
        if since:
            params["since"] = since
        if until:
            params["until"] = until

        response = self._request(
            "get",
            f"/repos/{self.owner}/{self.repo}/issues",
            params=params,
        )
        return self._parse_models(response, Issue, "issue listing")

    def close(self) -> None:
        """Close the HTTP client session."""
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def get_github_issue_tool() -> GitHubIssueTool:
    """Factory function to get a configured GitHubIssueTool instance."""
    return GitHubIssueTool()
