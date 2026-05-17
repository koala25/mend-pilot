"""Per-repo conventions cache."""

from __future__ import annotations

import json
import subprocess
import tomllib
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

CACHE_ROOT = Path("/data/repo_context")
CACHE_TTL = timedelta(days=7)


Extractor = Callable[[str, str], Awaitable[tuple[list[str], list[str], list[str], str]]]


@dataclass(frozen=True)
class RepoContext:
    repo_owner: str
    repo_name: str
    head_sha: str
    contributing_md: str | None
    readme_summary: str
    pr_template: str | None
    ruff_config: dict[str, Any] = field(default_factory=dict)
    mypy_config: dict[str, Any] = field(default_factory=dict)
    pre_commit_hooks: list[str] = field(default_factory=list)
    test_command: str = "pytest"
    style_notes: list[str] = field(default_factory=list)
    test_patterns: list[str] = field(default_factory=list)
    pr_norms: list[str] = field(default_factory=list)
    sample_test_excerpt: str = ""


async def load_repo_context(
    repo_path: Path,
    owner: str,
    name: str,
    *,
    extractor: Extractor | None = None,
) -> RepoContext:
    """Load or build the repo context. `extractor` may produce richer style notes via LLM."""
    head_sha = _git_head_sha(repo_path)
    cache_dir = CACHE_ROOT / owner / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{head_sha[:12]}.json"

    if cache_file.exists():
        age = datetime.now(UTC) - datetime.fromtimestamp(cache_file.stat().st_mtime, UTC)
        if age < CACHE_TTL:
            return RepoContext(**json.loads(cache_file.read_text()))

    contributing = _read_first(
        repo_path,
        [
            "CONTRIBUTING.md",
            ".github/CONTRIBUTING.md",
            "docs/contributing.md",
            "docs/en/docs/contributing.md",
        ],
    )
    pr_template = _read_first(
        repo_path,
        [
            ".github/PULL_REQUEST_TEMPLATE.md",
            ".github/pull_request_template.md",
            "PULL_REQUEST_TEMPLATE.md",
        ],
    )
    readme_raw = _read_first(repo_path, ["README.md", "README.rst"]) or ""
    pyproject = _parse_toml(repo_path / "pyproject.toml")
    pre_commit_hooks = _parse_pre_commit(repo_path / ".pre-commit-config.yaml")
    sample = _pick_sample_test(repo_path)

    style_notes: list[str] = []
    test_patterns: list[str] = []
    pr_norms: list[str] = []
    readme_summary = readme_raw[:1000]

    if extractor is not None:
        try:
            extracted = await extractor(contributing or "", readme_raw)
            style_notes, test_patterns, pr_norms, readme_summary = extracted
        except Exception:
            pass  # best-effort; fall back to defaults

    ctx = RepoContext(
        repo_owner=owner,
        repo_name=name,
        head_sha=head_sha,
        contributing_md=_truncate(contributing, 24000),
        readme_summary=readme_summary,
        pr_template=pr_template,
        ruff_config=pyproject.get("tool", {}).get("ruff") or {},
        mypy_config=pyproject.get("tool", {}).get("mypy") or {},
        pre_commit_hooks=pre_commit_hooks,
        test_command="pytest",
        style_notes=style_notes,
        test_patterns=test_patterns,
        pr_norms=pr_norms,
        sample_test_excerpt=sample,
    )
    cache_file.write_text(json.dumps(asdict(ctx), default=str, indent=2))
    return ctx


def _git_head_sha(repo_path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _read_first(root: Path, candidates: list[str]) -> str | None:
    for rel in candidates:
        p = root / rel
        if p.is_file():
            return p.read_text(errors="replace")
    return None


def _parse_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError:
        return {}


def _parse_pre_commit(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError:
        return []
    return [h["id"] for repo in data.get("repos", []) for h in repo.get("hooks", []) if "id" in h]


def _pick_sample_test(repo_path: Path) -> str:
    tests_dir = repo_path / "tests"
    if not tests_dir.exists():
        return ""
    candidates = [
        p
        for p in tests_dir.rglob("test_*.py")
        if p.name != "conftest.py" and p.stat().st_size < 50_000
    ]
    if not candidates:
        return ""
    pick = max(candidates, key=lambda p: p.stat().st_size)
    lines = pick.read_text(errors="replace").splitlines()[:80]
    return f"# {pick.relative_to(repo_path)}\n" + "\n".join(lines)


def _truncate(s: str | None, n: int) -> str | None:
    if s is None:
        return None
    return s if len(s) <= n else s[:n] + "\n... [truncated]"
