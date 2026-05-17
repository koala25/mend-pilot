"""Modal worker entry: pre-process guards → graph invoke → post-process."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import structlog

from ossagent.agent.context import load_repo_context
from ossagent.agent.context_extractor import make_extractor
from ossagent.agent.graph import build_graph
from ossagent.agent.state import AgentState, Classification
from ossagent.agent.tools import create_branch, shallow_clone_or_pull
from ossagent.config import load_models_config
from ossagent.db import Attempt, AttemptStatus, Database
from ossagent.github_client import GitHubClient
from ossagent.models import get_llm
from ossagent.telegram import TelegramBot

DAILY_CAP_USD = 5.00
MONTHLY_CAP_USD = 50.00
MAX_ATTEMPTS_PER_ISSUE = 3
RETRY_COOLDOWN = timedelta(days=2)
MAX_PER_REPO_PER_DAY = 5

log = structlog.get_logger()


async def process_issue(
    issue_url: str,
    classification: Classification,
    *,
    db_path: Path,
    config_dir: Path,
    our_login: str,
    data_dir: Path,
) -> None:
    db = Database(db_path)
    db.init_schema()

    if db.cost_month_to_date() > MONTHLY_CAP_USD * 0.95:
        log.warning("monthly_budget_threshold")
        return
    if db.cost_today() > DAILY_CAP_USD:
        log.info("daily_budget_breached")
        return

    prior = db.fetch_attempt_by_issue(issue_url)
    if prior:
        if prior.status in (
            AttemptStatus.DRAFTED_AWAITING_APPROVAL,
            AttemptStatus.PR_OPENED,
            AttemptStatus.MERGED,
        ):
            return
        if prior.attempt_count >= MAX_ATTEMPTS_PER_ISSUE:
            return
        if (datetime.now(UTC) - prior.started_at) < RETRY_COOLDOWN:
            return

    gh = GitHubClient()
    owner, name, number = _parse_issue_url(issue_url)
    issue = await gh.fetch_issue(owner, name, number)
    if issue.state == "closed" or issue.assignee is not None:
        return

    if db.repo_attempts_today(owner, name) >= MAX_PER_REPO_PER_DAY:
        return

    attempt_id = uuid4().hex
    db.record_attempt(
        Attempt(
            attempt_id=attempt_id,
            issue_url=issue_url,
            repo_owner=owner,
            repo_name=name,
            classification=classification,
            status=AttemptStatus.IN_PROGRESS,
            started_at=datetime.now(UTC),
            attempt_count=(prior.attempt_count + 1) if prior else 1,
        )
    )

    repo_path = data_dir / "repos" / owner / name
    shallow_clone_or_pull(f"https://github.com/{owner}/{name}.git", repo_path)
    create_branch(repo_path, f"fix/issue-{number}-{attempt_id[:8]}")

    models = load_models_config(config_dir / "models.yaml")
    extractor_llm = get_llm("planner", config=models)
    extractor = make_extractor(extractor_llm)
    repo_context = await load_repo_context(repo_path, owner, name, extractor=extractor)

    initial_state: AgentState = {
        "issue_url": issue_url,
        "repo_url": f"https://github.com/{owner}/{name}",
        "classification": classification,
        "attempt_id": attempt_id,
        "issue": issue,
        "repo_path": repo_path,
        "repo_context": repo_context,
        "retry_count": 0,
        "style_retry_count": 0,
        "cost_so_far": 0.0,
    }

    llms = {
        role: get_llm(role, config=models)
        for role in ("planner", "locator", "implementer", "tester", "critic", "pr_writer")
    }
    telegram_bot = TelegramBot()
    graph = build_graph(
        llms=llms,
        telegram_bot=telegram_bot,
        our_login=our_login,
        checkpoint_db=data_dir / "checkpoints.db",
    )
    try:
        final_state = await graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": attempt_id}, "recursion_limit": 50},
        )
    except Exception as e:
        log.exception("graph_failed", attempt_id=attempt_id, error=str(e))
        return

    if final_state.get("pr_metadata"):
        log.info("draft_ready", attempt_id=attempt_id)
        _update_status(db_path, attempt_id, AttemptStatus.DRAFTED_AWAITING_APPROVAL)
    else:
        log.info(
            "skipped",
            attempt_id=attempt_id,
            reason=final_state.get("skip_reason"),
        )
        _update_status(db_path, attempt_id, AttemptStatus.SKIPPED)


def _parse_issue_url(url: str) -> tuple[str, str, int]:
    parts = url.rstrip("/").split("/")
    return parts[-4], parts[-3], int(parts[-1])


def _update_status(db_path: Path, attempt_id: str, status: AttemptStatus) -> None:
    with sqlite3.connect(db_path) as c:
        c.execute(
            "UPDATE attempts SET status = ? WHERE attempt_id = ?",
            (status.value, attempt_id),
        )
