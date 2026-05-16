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
        return

    fits_sent = 0
    max_seen = last_seen or 0
    for issue in issues:
        max_seen = max(max_seen, issue.number)
        verdict = await triager.classify(issue)
        log.info(
            "triaged",
            issue=issue.number,
            fit=verdict.fit,
            conf=verdict.confidence,
            classification=verdict.classification,
        )
        if verdict.fit and verdict.confidence >= 0.7:
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

    db.set_last_seen_issue(repo.owner, repo.name, max_seen)
    log.info("polled", repo=f"{repo.owner}/{repo.name}", new_issues=len(issues), fits=fits_sent)


def _since_from_last_seen(last_seen: int | None) -> str:
    delta = timedelta(hours=24) if last_seen is None else timedelta(hours=1)
    return (datetime.now(UTC) - delta).strftime("%Y-%m-%dT%H:%M:%S")
