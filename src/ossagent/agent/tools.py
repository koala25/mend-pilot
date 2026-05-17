"""Deterministic helpers: git, diff application, ripgrep, file windowing."""

from __future__ import annotations

import subprocess
from pathlib import Path


def git(*args: str, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=check,
    )


def shallow_clone_or_pull(repo_url: str, dest: Path) -> Path:
    if dest.exists() and (dest / ".git").exists():
        git("fetch", "--depth", "1", "origin", cwd=dest)
        git("reset", "--hard", "origin/HEAD", cwd=dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(dest)],
            check=True,
        )
    return dest


def create_branch(repo_path: Path, branch_name: str) -> None:
    git("checkout", "-B", branch_name, cwd=repo_path)


def apply_unified_diff(repo_path: Path, diff_text: str) -> None:
    proc = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        cwd=repo_path,
        input=diff_text,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git apply failed:\n{proc.stdout}\n{proc.stderr}")


def stage_and_commit(repo_path: Path, message: str) -> None:
    git("add", "-A", cwd=repo_path)
    git("commit", "-m", message, cwd=repo_path)


def ripgrep(repo_path: Path, pattern: str, *, max_results: int = 50) -> list[str]:
    try:
        out = subprocess.run(
            ["rg", "--no-heading", "-n", pattern, str(repo_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        out = subprocess.run(
            ["grep", "-rnE", pattern, str(repo_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    return (out.stdout or "").splitlines()[:max_results]


def read_file_window(
    repo_path: Path,
    rel_path: str,
    line: int,
    *,
    before: int = 30,
    after: int = 60,
) -> str:
    p = repo_path / rel_path
    lines = p.read_text(errors="replace").splitlines()
    start = max(0, line - before - 1)
    end = min(len(lines), line + after)
    return "\n".join(f"{i + 1:5}| {lines[i]}" for i in range(start, end))
