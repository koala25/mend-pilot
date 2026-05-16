"""Modal application: schedules and entry points."""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("ossagent")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject.toml")
    .apt_install("git")
    # Modal 1.4+: bake the config dir into the image instead of using Mount.
    .add_local_dir("config", remote_path="/app/config")
)

vol = modal.Volume.from_name("ossagent-data", create_if_missing=True)
secrets = [modal.Secret.from_name("ossagent-secrets")]

CONFIG_MOUNT_PATH = Path("/app/config")
DB_PATH = Path("/data/attempts.db")


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
    """Scheduled tick: poll all watched repos, triage, notify."""
    import structlog

    from ossagent.config import load_models_config, load_watched_repos_config
    from ossagent.db import Database
    from ossagent.github_client import GitHubClient
    from ossagent.models import get_llm
    from ossagent.scheduler import poll_one_repo
    from ossagent.telegram import TelegramBot
    from ossagent.triager import Triager

    log = structlog.get_logger()

    models = load_models_config(CONFIG_MOUNT_PATH / "models.yaml")
    repos = load_watched_repos_config(CONFIG_MOUNT_PATH / "watched_repos.yaml")

    db = Database(DB_PATH)
    db.init_schema()

    gh = GitHubClient()
    triager = Triager(llm=get_llm("triager", config=models))
    telegram = TelegramBot()

    for repo in repos:
        try:
            await poll_one_repo(repo, gh=gh, triager=triager, telegram=telegram, db=db)
        except Exception as e:
            # Don't let one repo's failure stop the others.
            log.exception("poll_failed", repo=f"{repo.owner}/{repo.name}", error=str(e))


@app.local_entrypoint()  # type: ignore[misc]  # modal decorator is untyped
def main() -> None:
    """Manual trigger for testing: `modal run src/ossagent/app.py`."""
    poll_repos.remote()
