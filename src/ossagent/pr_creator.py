"""Push branch and open a GitHub draft PR via gh CLI."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from pathlib import Path

import structlog

from ossagent.agent.tools import git, stage_and_commit
from ossagent.db import AttemptStatus

log = structlog.get_logger()


def push_and_open_pr_from_attempt(
    *,
    attempt_id: str,
    data_dir: Path,
    db_path: Path,
) -> str | None:
    """Look up the attempt's prepared branch + PR metadata, push, open draft PR."""
    with sqlite3.connect(db_path) as c:
        c.row_factory = sqlite3.Row
        row = c.execute(
            "SELECT issue_url, repo_owner, repo_name FROM attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
    if row is None:
        log.warning("attempt_not_found", attempt_id=attempt_id)
        return None

    owner, name = row["repo_owner"], row["repo_name"]
    repo_path = data_dir / "repos" / owner / name

    branches = (
        subprocess.run(
            ["git", "branch", "--list", f"fix/issue-*{attempt_id[:8]}*"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .splitlines()
    )
    if not branches:
        log.warning("branch_not_found", attempt_id=attempt_id)
        return None
    branch = branches[0].strip().lstrip("* ").strip()

    sidecar = data_dir / "drafts" / attempt_id / "pr.json"
    meta = json.loads(sidecar.read_text())
    pr_title = meta["title"]
    pr_body = meta["body"]
    base_branch = meta["base_branch"]

    # The agent's enforce_style + add_test nodes left changes in the working
    # tree. Commit them now using the PR title as the commit message — this
    # makes the branch fast-forward-able and the eventual squash-merge clean.
    stage_and_commit(repo_path, pr_title)

    git("push", "-u", "origin", branch, cwd=repo_path)

    upstream = f"{owner}/{name}"
    proc = subprocess.run(
        [
            "gh",
            "pr",
            "create",
            "--draft",
            "--title",
            pr_title,
            "--body",
            pr_body,
            "--base",
            base_branch,
            "--repo",
            upstream,
        ],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ},
    )
    pr_url = (proc.stdout or "").strip().splitlines()[-1]
    with sqlite3.connect(db_path) as c:
        c.execute(
            "UPDATE attempts SET status = ?, pr_url = ? WHERE attempt_id = ?",
            (AttemptStatus.PR_OPENED.value, pr_url, attempt_id),
        )
    log.info("pr_opened", attempt_id=attempt_id, url=pr_url)
    return pr_url
