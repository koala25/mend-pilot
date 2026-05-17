"""Modal application: schedules, worker, webhook, PR opener, crons."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import modal

from ossagent.agent.state import Classification

app = modal.App("ossagent")

# Install gh from the official tarball — the Debian-slim apt index doesn't
# include the GitHub CLI without adding the upstream repo, and the tarball is
# the lightest reliable path.
GH_VERSION = "2.55.0"
INSTALL_GH = (
    f"curl -sSL https://github.com/cli/cli/releases/download/"
    f"v{GH_VERSION}/gh_{GH_VERSION}_linux_amd64.tar.gz | tar -xz -C /tmp && "
    f"mv /tmp/gh_{GH_VERSION}_linux_amd64/bin/gh /usr/local/bin/gh && "
    f"chmod +x /usr/local/bin/gh"
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ripgrep", "curl", "ca-certificates")
    .run_commands(INSTALL_GH)
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_dir("config", remote_path="/app/config")
    .add_local_python_source("ossagent")
)

vol = modal.Volume.from_name("ossagent-data", create_if_missing=True)
secrets = [modal.Secret.from_name("ossagent-secrets")]

CONFIG_MOUNT = Path("/app/config")
DATA_DIR = Path("/data")
DB_PATH = DATA_DIR / "attempts.db"
OUR_LOGIN = "koala25"


@app.function(  # type: ignore[misc]  # modal decorator is untyped
    image=image,
    schedule=modal.Period(minutes=30),
    cpu=0.5,
    memory=512,
    timeout=300,
    volumes={"/data": vol},
    secrets=secrets,
)
async def poll_repos() -> None:
    import structlog

    from ossagent.config import load_models_config, load_watched_repos_config
    from ossagent.db import Database
    from ossagent.github_client import GitHubClient
    from ossagent.models import get_llm
    from ossagent.scheduler import _since_from_last_seen
    from ossagent.telegram import TelegramBot, TriageNotification
    from ossagent.triager import Classification as TriageClass
    from ossagent.triager import Triager

    log = structlog.get_logger()

    models = load_models_config(CONFIG_MOUNT / "models.yaml")
    repos = load_watched_repos_config(CONFIG_MOUNT / "watched_repos.yaml")
    db = Database(DB_PATH)
    db.init_schema()
    gh = GitHubClient()
    triager = Triager(llm=get_llm("triager", config=models))
    telegram = TelegramBot()

    for repo in repos:
        last = db.get_last_seen_issue(repo.owner, repo.name)
        try:
            issues = await gh.fetch_new_issues(
                repo.owner,
                repo.name,
                since=_since_from_last_seen(last),
            )
        except Exception as e:
            log.exception("fetch_failed", repo=f"{repo.owner}/{repo.name}", error=str(e))
            continue
        max_seen = last or 0
        for issue in issues:
            max_seen = max(max_seen, issue.number)
            v = await triager.classify(issue)
            log.info(
                "triaged",
                issue=issue.number,
                fit=v.fit,
                conf=v.confidence,
                cls=v.classification,
            )
            if v.fit and v.confidence >= 0.7 and v.classification != TriageClass.UNCLASSIFIED:
                process_issue_fn.spawn(
                    issue_url=issue.html_url,
                    classification=v.classification.value,
                )
                await telegram.send_triage_notification(
                    TriageNotification(
                        issue_url=issue.html_url,
                        issue_title=issue.title,
                        classification=v.classification.value,
                        confidence=v.confidence,
                        reason=v.reason,
                    )
                )
        db.set_last_seen_issue(repo.owner, repo.name, max_seen)


@app.function(  # type: ignore[misc]  # modal decorator is untyped
    image=image,
    cpu=2,
    memory=4096,
    timeout=600,
    volumes={"/data": vol},
    secrets=secrets,
)
async def process_issue_fn(issue_url: str, classification: str) -> None:
    from ossagent.worker import process_issue

    await process_issue(
        issue_url,
        cast(Classification, classification),
        db_path=DB_PATH,
        config_dir=CONFIG_MOUNT,
        our_login=OUR_LOGIN,
        data_dir=DATA_DIR,
    )


@app.function(  # type: ignore[misc]  # modal decorator is untyped
    image=image,
    cpu=1,
    memory=1024,
    timeout=300,
    volumes={"/data": vol},
    secrets=secrets,
)
def push_and_open_pr_fn(attempt_id: str) -> None:
    from ossagent.pr_creator import push_and_open_pr_from_attempt

    push_and_open_pr_from_attempt(
        attempt_id=attempt_id,
        data_dir=DATA_DIR,
        db_path=DB_PATH,
    )


@app.function(  # type: ignore[misc]  # modal decorator is untyped
    image=image,
    cpu=0.5,
    memory=512,
    timeout=60,
    volumes={"/data": vol},
    secrets=secrets,
)
@modal.fastapi_endpoint(method="POST", label="telegram-webhook")  # type: ignore[misc]
async def telegram_webhook(payload: dict[str, Any]) -> dict[str, Any]:
    from ossagent.webhook import handle_telegram_callback

    return await handle_telegram_callback(
        payload,
        data_dir=DATA_DIR,
        db_path=DB_PATH,
    )


@app.function(  # type: ignore[misc]  # modal decorator is untyped
    image=image,
    schedule=modal.Period(days=1),
    cpu=0.5,
    memory=512,
    timeout=300,
    volumes={"/data": vol},
    secrets=secrets,
)
def daily_sync() -> None:
    from ossagent.crons import cleanup_stale_drafts, sync_pr_statuses

    sync_pr_statuses(DB_PATH)
    cleanup_stale_drafts(DB_PATH, DATA_DIR)


@app.local_entrypoint()  # type: ignore[misc]  # modal decorator is untyped
def main() -> None:
    poll_repos.remote()
