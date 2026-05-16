"""SQLite persistence: attempts, cost ledger, repo bookkeeping."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path


def _adapt_datetime(dt: datetime) -> str:
    """Store all datetimes as ISO-8601 UTC strings; reject naive datetimes."""
    if dt.tzinfo is None:
        raise ValueError("naive datetime cannot be stored; pass tz-aware UTC values")
    return dt.astimezone(UTC).isoformat()


def _convert_timestamp(value: bytes) -> datetime:
    """Read TIMESTAMP columns back as tz-aware UTC datetimes."""
    return datetime.fromisoformat(value.decode()).astimezone(UTC)


# Override the stdlib defaults, which are deprecated in 3.12 and TZ-lossy.
sqlite3.register_adapter(datetime, _adapt_datetime)
sqlite3.register_converter("TIMESTAMP", _convert_timestamp)


def _utc_today_start() -> datetime:
    """Midnight today in UTC (not local time)."""
    now = datetime.now(UTC)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _utc_month_start() -> datetime:
    """First moment of the current UTC month."""
    now = datetime.now(UTC)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


class AttemptStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    DRAFTED_AWAITING_APPROVAL = "drafted_awaiting_approval"
    PR_OPENED = "pr_opened"
    MERGED = "merged"
    REJECTED = "rejected"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class Attempt:
    attempt_id: str
    issue_url: str
    repo_owner: str
    repo_name: str
    classification: str
    status: AttemptStatus
    started_at: datetime
    attempt_count: int


@dataclass(frozen=True)
class CostLedgerEntry:
    attempt_id: str
    role: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    at: datetime


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, detect_types=sqlite3.PARSE_DECLTYPES)
        c.execute("PRAGMA foreign_keys = ON")
        c.row_factory = sqlite3.Row
        return c

    def init_schema(self) -> None:
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id     TEXT PRIMARY KEY,
                    issue_url      TEXT NOT NULL,
                    repo_owner     TEXT NOT NULL,
                    repo_name      TEXT NOT NULL,
                    classification TEXT NOT NULL,
                    status         TEXT NOT NULL,
                    started_at     TIMESTAMP NOT NULL,
                    attempt_count  INTEGER NOT NULL DEFAULT 1,
                    pr_url         TEXT
                );
                CREATE INDEX IF NOT EXISTS ix_attempts_repo
                    ON attempts(repo_owner, repo_name, started_at);
                CREATE INDEX IF NOT EXISTS ix_attempts_issue_url
                    ON attempts(issue_url);

                CREATE TABLE IF NOT EXISTS cost_ledger (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    attempt_id    TEXT NOT NULL,
                    role          TEXT NOT NULL,
                    input_tokens  INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    cost_usd      REAL NOT NULL,
                    at            TIMESTAMP NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_cost_at ON cost_ledger(at);

                CREATE TABLE IF NOT EXISTS repo_state (
                    repo_owner          TEXT NOT NULL,
                    repo_name           TEXT NOT NULL,
                    last_seen_issue     INTEGER,
                    PRIMARY KEY (repo_owner, repo_name)
                );
            """)

    def record_attempt(self, a: Attempt) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO attempts (attempt_id, issue_url, repo_owner, repo_name,
                    classification, status, started_at, attempt_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    a.attempt_id,
                    a.issue_url,
                    a.repo_owner,
                    a.repo_name,
                    a.classification,
                    a.status.value,
                    a.started_at,
                    a.attempt_count,
                ),
            )

    def fetch_attempt_by_issue(self, issue_url: str) -> Attempt | None:
        """Return the most recently started attempt for an issue, or None."""
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM attempts WHERE issue_url = ? ORDER BY started_at DESC LIMIT 1",
                (issue_url,),
            ).fetchone()
        if row is None:
            return None
        return Attempt(
            attempt_id=row["attempt_id"],
            issue_url=row["issue_url"],
            repo_owner=row["repo_owner"],
            repo_name=row["repo_name"],
            classification=row["classification"],
            status=AttemptStatus(row["status"]),
            started_at=row["started_at"],
            attempt_count=row["attempt_count"],
        )

    def repo_attempts_today(self, owner: str, name: str) -> int:
        today_start = _utc_today_start()
        with self._conn() as c:
            row = c.execute(
                """
                SELECT COUNT(*) AS n FROM attempts
                WHERE repo_owner = ? AND repo_name = ? AND started_at >= ?
            """,
                (owner, name, today_start),
            ).fetchone()
        return int(row["n"])

    def add_cost(self, e: CostLedgerEntry) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO cost_ledger (attempt_id, role, input_tokens,
                    output_tokens, cost_usd, at)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (e.attempt_id, e.role, e.input_tokens, e.output_tokens, e.cost_usd, e.at),
            )

    def cost_today(self) -> float:
        today_start = _utc_today_start()
        with self._conn() as c:
            row = c.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) AS s FROM cost_ledger WHERE at >= ?",
                (today_start,),
            ).fetchone()
        return float(row["s"])

    def cost_month_to_date(self) -> float:
        month_start = _utc_month_start()
        with self._conn() as c:
            row = c.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) AS s FROM cost_ledger WHERE at >= ?",
                (month_start,),
            ).fetchone()
        return float(row["s"])

    def set_last_seen_issue(self, owner: str, name: str, issue_number: int) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO repo_state (repo_owner, repo_name, last_seen_issue)
                VALUES (?, ?, ?)
                ON CONFLICT(repo_owner, repo_name) DO UPDATE SET
                    last_seen_issue = excluded.last_seen_issue
            """,
                (owner, name, issue_number),
            )

    def get_last_seen_issue(self, owner: str, name: str) -> int | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT last_seen_issue FROM repo_state WHERE repo_owner = ? AND repo_name = ?",
                (owner, name),
            ).fetchone()
        return None if row is None else row["last_seen_issue"]
