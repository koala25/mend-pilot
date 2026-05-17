"""Type definitions for the LangGraph agent state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

from ossagent.github_client import Issue

if TYPE_CHECKING:
    from ossagent.agent.context import RepoContext


@dataclass(frozen=True)
class PlanStep:
    number: int
    description: str


@dataclass(frozen=True)
class TargetFile:
    path: str
    line_range: tuple[int, int]
    why: str


@dataclass(frozen=True)
class TestRunResult:
    passed: bool
    failed_tests: list[str]
    stdout_tail: str
    stderr_tail: str


@dataclass(frozen=True)
class StyleViolation:
    tool: str
    file: str
    message: str


@dataclass(frozen=True)
class PRMetadata:
    title: str
    body: str
    branch_name: str
    head_owner: str
    base_branch: str


CriticVerdict = Literal["PASS", "RETRY", "ABORT"]
Classification = Literal["TYPO", "DEPRECATION", "TEST_GAP", "BUG_FIX"]


class AgentState(TypedDict, total=False):
    # Input
    issue_url: str
    repo_url: str
    classification: Classification
    attempt_id: str

    # Loaded
    issue: Issue
    repo_path: Path
    repo_context: RepoContext

    # Planning / locating
    plan: list[PlanStep]
    target_files: list[TargetFile]

    # Reproduction (BUG_FIX)
    failing_test_path: str | None
    failing_test_output: str | None

    # Implementation
    patch: str

    # Validation
    style_violations: list[StyleViolation]
    style_retry_count: int
    test_run: TestRunResult
    critic_verdict: CriticVerdict
    critic_reasoning: str
    confidence: float

    # Bookkeeping
    retry_count: int
    cost_so_far: float
    skip_reason: str

    # Output
    pr_metadata: PRMetadata
