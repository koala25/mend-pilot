"""Telegram webhook handler."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import structlog

from ossagent.db import AttemptStatus

log = structlog.get_logger()


async def handle_telegram_callback(
    payload: dict[str, Any],
    *,
    data_dir: Path,
    db_path: Path,
) -> dict[str, Any]:
    callback = payload.get("callback_query")
    if not callback:
        return {"ok": True}
    sender_id = str(callback["from"]["id"])
    expected_id = os.environ["TELEGRAM_USER_ID"]
    if sender_id != expected_id:
        return {"ok": False, "error": "unauthorized"}
    data = callback.get("data", "")
    try:
        attempt_id, action = data.split(":", 1)
    except ValueError:
        return {"ok": False, "error": "bad_callback_data"}
    log.info("webhook_action", attempt_id=attempt_id, action=action)
    if action == "approve":
        # Lazy import to avoid pulling Modal into webhook unit tests, and to
        # break the circular link with src/ossagent/app.py which imports this
        # module in its FastAPI route.
        from ossagent.app import push_and_open_pr_fn  # type: ignore[attr-defined]

        push_and_open_pr_fn.spawn(attempt_id=attempt_id)
    elif action == "reject":
        with sqlite3.connect(db_path) as c:
            c.execute(
                "UPDATE attempts SET status = ? WHERE attempt_id = ?",
                (AttemptStatus.REJECTED.value, attempt_id),
            )
    else:
        return {"ok": False, "error": "unknown_action"}
    return {"ok": True}
