"""Graph state and structured artifacts passed between agents."""

from functools import lru_cache
from typing import Literal, TypedDict, get_type_hints

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
    revision_target: Literal["code", "tests"] = Field(
        default="code",
        description="What needs fixing: 'code' for implementation issues, 'tests' for bad test cases",
    )


class TeamState(TypedDict, total=False):
    """State flowing through the Planner → Coder → Tester → Reviewer graph.

    The task, the plan and the human feedback are transient agent context:
    they live on the run's GitHub issue and are fetched from there whenever
    a node needs them, so the checkpoint never grows with the size of the
    plan. The generated code *is* checkpointed (``code``) so a paused run
    can be inspected and resumed without re-running the Coder.
    """

    # Essential checkpointed state
    issue_id: int
    code: CodeResult
    test_report: TestReport
    review: Review
    approved: bool
    # The human's verdict on the plan. It is persisted so the conditional
    # edge after approve_plan can route on it; the plan itself stays
    # transient and is re-read from the issue.
    plan_approved: bool
    # The revision counter. It is a counter rather than context, and it must
    # survive a restart — otherwise the max_iterations guard would reset
    # every time the run is resumed.
    iteration: int


@lru_cache(maxsize=1)
def msgpack_allowlist() -> tuple[tuple[str, str], ...]:
    """(module, ClassName) for every pydantic model in TeamState.

    The checkpointer serializes state to msgpack; langgraph requires pydantic
    types to be explicitly allowlisted for deserialization.
    """
    return tuple(
        (ann.__module__, ann.__name__)
        for ann in get_type_hints(TeamState).values()
        if isinstance(ann, type) and issubclass(ann, BaseModel)
    )
