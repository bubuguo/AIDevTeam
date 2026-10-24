"""Graph state and structured artifacts passed between agents."""

from typing import TypedDict

from pydantic import BaseModel, Field


class FileSpec(BaseModel):
    """A file the Planner wants implemented (no content yet)."""

    path: str = Field(description="Relative file path inside the workspace, e.g. 'app.py'")
    description: str = Field(description="What this file contains and its responsibility")


class Plan(BaseModel):
    """The Planner's build plan."""

    app_name: str = Field(description="Short name of the CLI application")
    summary: str = Field(description="One-paragraph description of what the app does")
    cli_commands: list[str] = Field(
        default_factory=list, description="Example CLI invocations, e.g. 'python app.py convert in.csv out.json'"
    )
    files: list[FileSpec] = Field(description="Files to create")


class GeneratedFile(BaseModel):
    """A complete file produced by the Coder or Tester."""

    path: str = Field(description="Relative file path inside the workspace")
    content: str = Field(description="Full file content")


class CodeResult(BaseModel):
    """The Coder's output: all implemented files."""

    files: list[GeneratedFile] = Field(description="Every file, with FULL content")
    notes: str = Field(default="", description="Implementation notes for the team")


class TestSuite(BaseModel):
    """The Tester's generated test files."""

    files: list[GeneratedFile] = Field(description="Test files named test_*.py, with FULL content")


class TestReport(BaseModel):
    """Real execution results from running the test suite."""

    passed: bool = Field(description="True when the whole suite exited successfully")
    summary: str = Field(description="One-line human summary of the results")
    output: str = Field(default="", description="Truncated pytest / smoke-test output")


class Review(BaseModel):
    """The Reviewer's verdict."""

    approved: bool = Field(description="True when the submission is ready to ship")
    feedback: list[str] = Field(
        default_factory=list, description="Concrete, actionable fixes required (empty when approved)"
    )


class TeamState(TypedDict, total=False):
    """State flowing through the Planner → Coder → Tester → Reviewer graph."""

    task: str
    workspace: str
    plan: Plan
    code: CodeResult
    test_report: TestReport
    review: Review
    feedback: list[str]
    iteration: int
    approved: bool
