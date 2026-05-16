"""Per-tick orchestration: poll a repo, triage new issues, notify."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import structlog

from ossagent.config import WatchedRepo
from ossagent.db import Database
from ossagent.github_client import GitHubClient
from ossagent.telegram import TelegramBot, TriageNotification
from ossagent.triager import Triager

log = structlog.get_logger()


async def poll_one_repo(
    repo: WatchedRepo,
    *,
    gh: GitHubClient,
    triager: Triager,
    telegram: TelegramBot,
    db: Database,
) -> None:
    last_seen = db.get_last_seen_issue(repo.owner, repo.name)
    since = _since_from_last_seen(last_seen)
    log.info("polling", repo=f"{repo.owner}/{repo.name}", since=since)

    issues = await gh.fetch_new_issues(repo.owner, repo.name, since=since)
    if not issues:
        log.info("polled", repo=f"{repo.owner}/{repo.name}", new_issues=0, fits=0)
        return

    fits_sent = 0
    skipped_seen = 0
    max_seen = last_seen or 0
    for issue in issues:
        # GitHub's `since` is keyed on updated_at, so re-commented old issues
        # come back every tick. Skip anything at or below our watermark.
        if last_seen is not None and issue.number <= last_seen:
            skipped_seen += 1
            continue
        max_seen = max(max_seen, issue.number)
        try:
            verdict = await triager.classify(issue)
        except Exception:
            log.exception("triage_failed", issue=issue.number)
            continue
        log.info(
            "triaged",
            issue=issue.number,
            fit=verdict.fit,
            conf=verdict.confidence,
            classification=verdict.classification.value,
        )
        if verdict.fit and verdict.confidence >= 0.7:
            try:
                await telegram.send_triage_notification(
                    TriageNotification(
                        issue_url=issue.html_url,
                        issue_title=issue.title,
                        classification=verdict.classification.value,
                        confidence=verdict.confidence,
                        reason=verdict.reason,
                    )
                )
                fits_sent += 1
            except Exception:
                log.exception("notify_failed", issue=issue.number)

    db.set_last_seen_issue(repo.owner, repo.name, max_seen)
    log.info(
        "polled",
        repo=f"{repo.owner}/{repo.name}",
        new_issues=len(issues),
        skipped_already_seen=skipped_seen,
        fits=fits_sent,
    )


def _since_from_last_seen(last_seen: int | None) -> str:
    delta = timedelta(hours=24) if last_seen is None else timedelta(hours=1)
    return (datetime.now(UTC) - delta).strftime("%Y-%m-%dT%H:%M:%S")
