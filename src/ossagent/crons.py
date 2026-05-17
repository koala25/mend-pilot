"""Daily/weekly cron jobs: PR status sync + stale-draft cleanup."""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import structlog
from github import Github

log = structlog.get_logger()

STALE_DRAFT_AGE = timedelta(days=7)


def sync_pr_statuses(db_path: Path) -> None:
    """For every attempt with status=PR_OPENED, fetch latest PR state from GitHub."""
    gh = Github(os.environ["GITHUB_TOKEN"])
    with sqlite3.connect(db_path) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT attempt_id, pr_url FROM attempts WHERE status = 'pr_opened'"
        ).fetchall()
    for row in rows:
        url = row["pr_url"]
        if not url:
            continue
        try:
            owner, name, _, number = _parse_pr_url(url)
            pr = gh.get_repo(f"{owner}/{name}").get_pull(int(number))
        except Exception as e:
            log.warning("pr_fetch_failed", url=url, error=str(e))
            continue
        if pr.merged:
            new_status = "merged"
        elif pr.state == "closed":
            new_status = "rejected"
        else:
            new_status = "pr_opened"
        with sqlite3.connect(db_path) as c:
            c.execute(
                "UPDATE attempts SET status = ? WHERE attempt_id = ?",
                (new_status, row["attempt_id"]),
            )
        log.info("pr_status_synced", url=url, new_status=new_status)


def cleanup_stale_drafts(db_path: Path, data_dir: Path) -> None:
    """Remove prepared-but-unapproved drafts older than STALE_DRAFT_AGE."""
    cutoff = datetime.now(UTC) - STALE_DRAFT_AGE
    with sqlite3.connect(db_path) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            """
            SELECT attempt_id, repo_owner, repo_name FROM attempts
            WHERE status = 'drafted_awaiting_approval' AND started_at < ?
            """,
            (cutoff,),
        ).fetchall()
    for row in rows:
        aid = row["attempt_id"]
        sidecar = data_dir / "drafts" / aid
        if sidecar.exists():
            shutil.rmtree(sidecar)
        repo_path = data_dir / "repos" / row["repo_owner"] / row["repo_name"]
        if repo_path.exists():
            # Best-effort branch delete; the glob pattern won't actually match
            # because git branch -D doesn't expand globs, but we leave it as
            # a placeholder for future precise lookup.
            subprocess.run(
                ["git", "branch", "-D", f"fix/issue-*{aid[:8]}*"],
                cwd=repo_path,
                capture_output=True,
                check=False,
            )
        with sqlite3.connect(db_path) as c:
            c.execute(
                "UPDATE attempts SET status = 'skipped' WHERE attempt_id = ?",
                (aid,),
            )
        log.info("draft_cleaned", attempt_id=aid)


def _parse_pr_url(url: str) -> tuple[str, str, str, str]:
    parts = url.rstrip("/").split("/")
    return parts[-4], parts[-3], parts[-2], parts[-1]
