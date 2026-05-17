# Phase 1 — Skeleton Implementation Plan (no-TDD edition)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy a Modal app that polls watched OSS repos every 30 minutes, classifies new issues with a cheap LLM triager, persists results to SQLite, and sends Telegram notifications for fit issues. No agent execution yet — Phase 1 is plumbing only.

**Architecture:** A single Modal app with one scheduled function. Inside: a model-agnostic LLM factory (Kimi default, Anthropic-pluggable), a thin GitHub API client, an outbound-only Telegram bot wrapper, a cost telemetry SQLite store, and a deterministic Triager that returns structured JSON.

**Tech Stack:** Python 3.12 · uv · LangChain (`langchain-openai` for Moonshot compatibility) · Modal · httpx · SQLite (stdlib) · ruff · mypy · pre-commit

**Testing policy:** Tests are explicitly **deferred** to a Phase 1.5 hardening pass after the system runs end-to-end. Per-task verification is: file imports cleanly, lints clean, runs without error when invoked. No unit tests in this plan.

---

## File Structure

```
mend-pilot/
├── pyproject.toml                 # deps, ruff/mypy config
├── .pre-commit-config.yaml        # ruff + mypy gating
├── .gitignore
├── config/
│   ├── models.yaml                # role → provider/model mapping
│   └── watched_repos.yaml         # repos to poll
└── src/
    └── ossagent/
        ├── __init__.py
        ├── config.py              # load YAML configs
        ├── models.py              # get_llm(role) factory
        ├── telemetry.py           # CostTracker callback
        ├── db.py                  # SQLite schema + helpers
        ├── github_client.py       # fetch_new_issues, fetch_issue
        ├── telegram.py            # send_message (outbound only)
        ├── triager.py             # classify_issue → TriageVerdict
        ├── scheduler.py           # poll_one_repo orchestration
        └── app.py                 # Modal app entry
```

One file = one responsibility. Modal app code (`app.py`) stays thin — logic lives in `scheduler.py`.

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/ossagent/__init__.py`

- [ ] **Step 1: Create directory structure**

```bash
cd /Users/kbtg/codebase/personal/mend-pilot
mkdir -p src/ossagent config
touch src/ossagent/__init__.py
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "ossagent"
version = "0.1.0"
description = "Multi-agent OSS PR contributor bot"
requires-python = ">=3.12"
dependencies = [
    "modal>=0.66",
    "langchain>=0.3",
    "langchain-openai>=0.2",
    "langchain-anthropic>=0.3",
    "httpx>=0.27",
    "pydantic>=2.9",
    "pyyaml>=6.0",
    "structlog>=24.4",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.7",
    "mypy>=1.13",
    "pre-commit>=4.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ossagent"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "RUF"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.12"
strict = true
ignore_missing_imports = true
```

- [ ] **Step 3: Write `.gitignore`**

```
__pycache__/
*.pyc
.venv/
.env
.modal/
.pytest_cache/
.mypy_cache/
.ruff_cache/
*.db
*.db-journal
dist/
build/
*.egg-info/
.coverage
.DS_Store
```

- [ ] **Step 4: Install deps with `uv`**

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

Expected: venv created at `.venv/`, all deps installed without errors.

- [ ] **Step 5: Verify import baseline**

```bash
.venv/bin/python -c "import ossagent; print('ok')"
.venv/bin/ruff check src
```

Expected: prints `ok`; ruff reports zero issues.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore src/
git commit -m "chore: scaffold pyproject + dirs"
```

---

## Task 2: YAML config loaders

**Files:**
- Create: `src/ossagent/config.py`, `config/models.yaml`, `config/watched_repos.yaml`

- [ ] **Step 1: Write `src/ossagent/config.py`**

```python
"""Load and validate YAML configuration files."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


REQUIRED_ROLES = (
    "triager", "planner", "locator", "implementer",
    "tester", "critic", "pr_writer",
)


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    api_base: str | None
    model: str
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class WatchedRepo:
    owner: str
    name: str
    default_branch: str


def load_models_config(path: Path) -> dict[str, ModelSpec]:
    raw = yaml.safe_load(path.read_text())
    defaults = raw.get("defaults", {})
    roles_raw = raw.get("roles", {})
    result: dict[str, ModelSpec] = {}
    for role in REQUIRED_ROLES:
        if role not in roles_raw:
            raise KeyError(f"Missing required role: {role}")
        merged = {**defaults, **roles_raw[role]}
        result[role] = ModelSpec(
            provider=merged["provider"],
            api_base=merged.get("api_base"),
            model=merged["model"],
            temperature=float(merged.get("temperature", 0.1)),
            max_tokens=int(merged["max_tokens"]),
        )
    return result


def load_watched_repos_config(path: Path) -> list[WatchedRepo]:
    raw = yaml.safe_load(path.read_text())
    return [
        WatchedRepo(
            owner=r["owner"], name=r["name"],
            default_branch=r.get("default_branch", "main"),
        )
        for r in raw["repos"]
    ]
```

- [ ] **Step 2: Write `config/models.yaml`**

```yaml
defaults:
  provider: moonshot
  api_base: https://api.moonshot.cn/v1
  model: kimi-latest       # PIN to exact identifier when you have it
  temperature: 0.1
  max_tokens: 2000

roles:
  triager:     { temperature: 0.0, max_tokens: 200 }
  planner:     { temperature: 0.2, max_tokens: 1500 }
  locator:     { temperature: 0.0, max_tokens: 800 }
  implementer: { temperature: 0.2, max_tokens: 4000 }
  tester:      { temperature: 0.1, max_tokens: 2000 }
  critic:      { temperature: 0.2, max_tokens: 1500 }
  pr_writer:   { temperature: 0.3, max_tokens: 1000 }
```

- [ ] **Step 3: Write `config/watched_repos.yaml`**

```yaml
repos:
  - owner: langchain-ai
    name: langchain
    default_branch: master
```

(Phase 1 starts with one repo. Add more in Phase 2.)

- [ ] **Step 4: Verify**

```bash
.venv/bin/python -c "
from pathlib import Path
from ossagent.config import load_models_config, load_watched_repos_config
models = load_models_config(Path('config/models.yaml'))
repos = load_watched_repos_config(Path('config/watched_repos.yaml'))
print(f'models: {len(models)} roles, repos: {len(repos)}')
"
```

Expected: `models: 7 roles, repos: 1`

- [ ] **Step 5: Commit**

```bash
git add config/ src/ossagent/config.py
git commit -m "feat(config): YAML loaders for models and repos"
```

---

## Task 3: Model factory

**Files:**
- Create: `src/ossagent/models.py`

- [ ] **Step 1: Write `src/ossagent/models.py`**

```python
"""Provider-agnostic LLM factory."""
from __future__ import annotations
import os
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from ossagent.config import ModelSpec


def get_llm(role: str, *, config: dict[str, ModelSpec]) -> BaseChatModel:
    spec = config[role]
    if spec.provider in ("moonshot", "openai"):
        api_key_env = {
            "moonshot": "MOONSHOT_API_KEY",
            "openai": "OPENAI_API_KEY",
        }[spec.provider]
        return ChatOpenAI(
            base_url=spec.api_base,
            api_key=os.environ[api_key_env],
            model=spec.model,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
        )
    if spec.provider == "anthropic":
        return ChatAnthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model=spec.model,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
        )
    raise ValueError(f"Unknown provider: {spec.provider}")
```

- [ ] **Step 2: Verify import**

```bash
.venv/bin/python -c "from ossagent.models import get_llm; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/ossagent/models.py
git commit -m "feat(models): provider-agnostic LLM factory"
```

---

## Task 4: SQLite schema + helpers

**Files:**
- Create: `src/ossagent/db.py`

- [ ] **Step 1: Write `src/ossagent/db.py`**

```python
"""SQLite persistence: attempts, cost ledger, repo bookkeeping."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, UTC
from enum import StrEnum
from pathlib import Path
import sqlite3


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
            c.execute("""
                INSERT INTO attempts (attempt_id, issue_url, repo_owner, repo_name,
                    classification, status, started_at, attempt_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (a.attempt_id, a.issue_url, a.repo_owner, a.repo_name,
                  a.classification, a.status.value, a.started_at, a.attempt_count))

    def fetch_attempt_by_issue(self, issue_url: str) -> Attempt | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM attempts WHERE issue_url = ? ORDER BY started_at DESC LIMIT 1",
                (issue_url,),
            ).fetchone()
        if row is None:
            return None
        return Attempt(
            attempt_id=row["attempt_id"], issue_url=row["issue_url"],
            repo_owner=row["repo_owner"], repo_name=row["repo_name"],
            classification=row["classification"],
            status=AttemptStatus(row["status"]),
            started_at=row["started_at"], attempt_count=row["attempt_count"],
        )

    def repo_attempts_today(self, owner: str, name: str) -> int:
        today_start = datetime.combine(date.today(), datetime.min.time(), UTC)
        with self._conn() as c:
            row = c.execute("""
                SELECT COUNT(*) AS n FROM attempts
                WHERE repo_owner = ? AND repo_name = ? AND started_at >= ?
            """, (owner, name, today_start)).fetchone()
        return int(row["n"])

    def add_cost(self, e: CostLedgerEntry) -> None:
        with self._conn() as c:
            c.execute("""
                INSERT INTO cost_ledger (attempt_id, role, input_tokens,
                    output_tokens, cost_usd, at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (e.attempt_id, e.role, e.input_tokens, e.output_tokens,
                  e.cost_usd, e.at))

    def cost_today(self) -> float:
        today_start = datetime.combine(date.today(), datetime.min.time(), UTC)
        with self._conn() as c:
            row = c.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) AS s FROM cost_ledger WHERE at >= ?",
                (today_start,),
            ).fetchone()
        return float(row["s"])

    def cost_month_to_date(self) -> float:
        month_start = datetime(date.today().year, date.today().month, 1,
                               tzinfo=UTC)
        with self._conn() as c:
            row = c.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) AS s FROM cost_ledger WHERE at >= ?",
                (month_start,),
            ).fetchone()
        return float(row["s"])

    def set_last_seen_issue(self, owner: str, name: str, issue_number: int) -> None:
        with self._conn() as c:
            c.execute("""
                INSERT INTO repo_state (repo_owner, repo_name, last_seen_issue)
                VALUES (?, ?, ?)
                ON CONFLICT(repo_owner, repo_name) DO UPDATE SET
                    last_seen_issue = excluded.last_seen_issue
            """, (owner, name, issue_number))

    def get_last_seen_issue(self, owner: str, name: str) -> int | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT last_seen_issue FROM repo_state WHERE repo_owner = ? AND repo_name = ?",
                (owner, name),
            ).fetchone()
        return None if row is None else row["last_seen_issue"]
```

- [ ] **Step 2: Verify schema initializes**

```bash
.venv/bin/python -c "
from pathlib import Path
from ossagent.db import Database
db = Database(Path('/tmp/ossagent-verify.db'))
db.init_schema()
print('ok')
" && rm -f /tmp/ossagent-verify.db
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/ossagent/db.py
git commit -m "feat(db): SQLite schema for attempts, cost ledger, repo state"
```

---

## Task 5: Cost telemetry callback

**Files:**
- Create: `src/ossagent/telemetry.py`

- [ ] **Step 1: Write `src/ossagent/telemetry.py`**

```python
"""Cost-tracking LangChain callback that writes to the cost_ledger table."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, UTC
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from ossagent.db import CostLedgerEntry, Database


@dataclass(frozen=True)
class ModelPrice:
    input_per_1m: float
    output_per_1m: float


# Pin from provider docs at deploy time.
MODEL_PRICING: dict[str, ModelPrice] = {
    "kimi-latest":       ModelPrice(input_per_1m=0.30, output_per_1m=1.50),
    "claude-haiku-4-5":  ModelPrice(input_per_1m=0.80, output_per_1m=4.00),
    "claude-sonnet-4-6": ModelPrice(input_per_1m=3.00, output_per_1m=15.00),
    "claude-opus-4-7":   ModelPrice(input_per_1m=15.00, output_per_1m=75.00),
}


class CostTracker(BaseCallbackHandler):
    def __init__(self, db: Database, attempt_id: str, role: str, model: str) -> None:
        self.db = db
        self.attempt_id = attempt_id
        self.role = role
        self.model = model

    def on_llm_end(self, response: LLMResult, **kwargs: object) -> None:
        usage = self._extract_usage(response)
        if usage is None:
            return
        input_tokens, output_tokens = usage
        price = MODEL_PRICING.get(self.model)
        cost_usd = (
            0.0 if price is None else
            (input_tokens / 1_000_000) * price.input_per_1m
            + (output_tokens / 1_000_000) * price.output_per_1m
        )
        self.db.add_cost(CostLedgerEntry(
            attempt_id=self.attempt_id, role=self.role,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=cost_usd, at=datetime.now(UTC),
        ))

    @staticmethod
    def _extract_usage(response: LLMResult) -> tuple[int, int] | None:
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                um = getattr(msg, "usage_metadata", None)
                if um:
                    return int(um["input_tokens"]), int(um["output_tokens"])
        return None
```

- [ ] **Step 2: Verify import**

```bash
.venv/bin/python -c "from ossagent.telemetry import CostTracker, MODEL_PRICING; print(len(MODEL_PRICING), 'models priced')"
```

Expected: `4 models priced`.

- [ ] **Step 3: Commit**

```bash
git add src/ossagent/telemetry.py
git commit -m "feat(telemetry): cost-tracking LangChain callback"
```

---

## Task 6: GitHub client

**Files:**
- Create: `src/ossagent/github_client.py`

- [ ] **Step 1: Write `src/ossagent/github_client.py`**

```python
"""Thin async GitHub REST client."""
from __future__ import annotations
import os
from dataclasses import dataclass
import httpx


@dataclass(frozen=True)
class Issue:
    number: int
    title: str
    body: str
    labels: list[str]
    author: str
    html_url: str
    state: str
    assignee: str | None
    comments: int
    created_at: str


class GitHubClient:
    BASE = "https://api.github.com"

    def __init__(self, token: str | None = None) -> None:
        self.token = token or os.environ["GITHUB_TOKEN"]

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def fetch_new_issues(self, owner: str, name: str, *, since: str) -> list[Issue]:
        url = f"{self.BASE}/repos/{owner}/{name}/issues"
        params = {
            "state": "open", "sort": "created", "direction": "desc",
            "per_page": "30", "since": since,
        }
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url, headers=self._headers(), params=params)
            r.raise_for_status()
            data = r.json()
        return [self._to_issue(d) for d in data if "pull_request" not in d]

    async def fetch_issue(self, owner: str, name: str, number: int) -> Issue:
        url = f"{self.BASE}/repos/{owner}/{name}/issues/{number}"
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url, headers=self._headers())
            r.raise_for_status()
            return self._to_issue(r.json())

    @staticmethod
    def _to_issue(d: dict) -> Issue:
        return Issue(
            number=d["number"], title=d["title"], body=d.get("body") or "",
            labels=[lbl["name"] for lbl in d.get("labels", [])],
            author=d["user"]["login"], html_url=d["html_url"],
            state=d["state"],
            assignee=(d.get("assignee") or {}).get("login"),
            comments=d.get("comments", 0), created_at=d["created_at"],
        )
```

- [ ] **Step 2: Verify import**

```bash
.venv/bin/python -c "from ossagent.github_client import GitHubClient; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/ossagent/github_client.py
git commit -m "feat(github): async client for issue listing and fetching"
```

---

## Task 7: Telegram outbound bot

**Files:**
- Create: `src/ossagent/telegram.py`

- [ ] **Step 1: Write `src/ossagent/telegram.py`**

```python
"""Outbound Telegram notifications. Webhook handling is added in Phase 2."""
from __future__ import annotations
import os
from dataclasses import dataclass
import httpx


@dataclass(frozen=True)
class TriageNotification:
    issue_url: str
    issue_title: str
    classification: str
    confidence: float
    reason: str


class TelegramBot:
    def __init__(self, token: str | None = None, user_id: str | None = None) -> None:
        self.token = token or os.environ["TELEGRAM_BOT_TOKEN"]
        self.user_id = user_id or os.environ["TELEGRAM_USER_ID"]

    async def send_triage_notification(self, n: TriageNotification) -> int:
        text = (
            f"🔍 New fit issue\n\n"
            f"<b>{_escape(n.issue_title)}</b>\n"
            f"{n.issue_url}\n\n"
            f"Classification: <b>{n.classification}</b>\n"
            f"Confidence: <b>{n.confidence:.2f}</b>\n"
            f"Reason: {_escape(n.reason)}"
        )
        return await self._send(text)

    async def _send(self, text: str) -> int:
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.user_id, "text": text,
            "parse_mode": "HTML", "disable_web_page_preview": True,
        }
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(url, json=payload)
            if r.status_code != 200:
                raise RuntimeError(f"Telegram API failed: {r.status_code} {r.text}")
            return int(r.json()["result"]["message_id"])


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
```

- [ ] **Step 2: Verify import**

```bash
.venv/bin/python -c "from ossagent.telegram import TelegramBot, TriageNotification; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/ossagent/telegram.py
git commit -m "feat(telegram): outbound triage-notification bot"
```

---

## Task 8: Triager — Stage 1 LLM classifier

**Files:**
- Create: `src/ossagent/triager.py`

- [ ] **Step 1: Write `src/ossagent/triager.py`**

```python
"""Stage 1 cheap LLM classifier. Decides if an issue is worth the heavy worker."""
from __future__ import annotations
import json
from dataclasses import dataclass
from enum import StrEnum
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from ossagent.github_client import Issue


class Classification(StrEnum):
    TYPO = "TYPO"
    DEPRECATION = "DEPRECATION"
    TEST_GAP = "TEST_GAP"
    BUG_FIX = "BUG_FIX"


@dataclass(frozen=True)
class TriageVerdict:
    fit: bool
    confidence: float
    classification: Classification
    reason: str


SYSTEM_PROMPT = """You are a strict triager for an automated open-source contribution bot.

You will be given a GitHub issue. Decide whether the issue is a good candidate
for an automated single-file, mechanical fix. Reject anything that requires:
- Multi-file refactoring
- Architectural decisions
- Feature work without a clear spec
- Vague bug reports with no repro

Reply with STRICT JSON only. No commentary.

Schema:
{
  "fit": true | false,
  "confidence": 0.0-1.0,
  "class": "TYPO" | "DEPRECATION" | "TEST_GAP" | "BUG_FIX",
  "reason": "<one short sentence>"
}
"""


class Triager:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    async def classify(self, issue: Issue) -> TriageVerdict:
        user_msg = (
            f"Title: {issue.title}\n\n"
            f"Body:\n{issue.body[:2000]}\n\n"
            f"Labels: {', '.join(issue.labels) or '(none)'}\n"
            f"Comments: {issue.comments}"
        )
        msg = await self.llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])
        return self._parse(msg.content)

    @staticmethod
    def _parse(text: str) -> TriageVerdict:
        try:
            data = json.loads(text.strip())
            return TriageVerdict(
                fit=bool(data["fit"]),
                confidence=float(data["confidence"]),
                classification=Classification(data["class"]),
                reason=str(data["reason"]),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return TriageVerdict(
                fit=False, confidence=0.0,
                classification=Classification.BUG_FIX,
                reason="Triager returned malformed JSON; rejecting by default.",
            )
```

- [ ] **Step 2: Verify import**

```bash
.venv/bin/python -c "from ossagent.triager import Triager, Classification, TriageVerdict; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/ossagent/triager.py
git commit -m "feat(triager): Stage 1 LLM classifier with strict JSON parsing"
```

---

## Task 9: Scheduler orchestration logic

**Files:**
- Create: `src/ossagent/scheduler.py`

- [ ] **Step 1: Write `src/ossagent/scheduler.py`**

```python
"""Per-tick orchestration: poll a repo, triage new issues, notify."""
from __future__ import annotations
from datetime import datetime, timedelta, UTC
import structlog
from ossagent.config import WatchedRepo
from ossagent.db import Database
from ossagent.github_client import GitHubClient
from ossagent.telegram import TelegramBot, TriageNotification
from ossagent.triager import Triager

log = structlog.get_logger()


async def poll_one_repo(
    repo: WatchedRepo, *,
    gh: GitHubClient, triager: Triager, telegram: TelegramBot, db: Database,
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
        log.info("triaged", issue=issue.number, fit=verdict.fit,
                 conf=verdict.confidence, classification=verdict.classification)
        if verdict.fit and verdict.confidence >= 0.7:
            await telegram.send_triage_notification(TriageNotification(
                issue_url=issue.html_url, issue_title=issue.title,
                classification=verdict.classification.value,
                confidence=verdict.confidence, reason=verdict.reason,
            ))
            fits_sent += 1

    db.set_last_seen_issue(repo.owner, repo.name, max_seen)
    log.info("polled", repo=f"{repo.owner}/{repo.name}",
             new_issues=len(issues), fits=fits_sent)


def _since_from_last_seen(last_seen: int | None) -> str:
    delta = timedelta(hours=24) if last_seen is None else timedelta(hours=1)
    return (datetime.now(UTC) - delta).strftime("%Y-%m-%dT%H:%M:%S")
```

- [ ] **Step 2: Verify import**

```bash
.venv/bin/python -c "from ossagent.scheduler import poll_one_repo; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/ossagent/scheduler.py
git commit -m "feat(scheduler): per-tick repo polling + triage + notify"
```

---

## Task 10: Modal app wiring

**Files:**
- Create: `src/ossagent/app.py`

- [ ] **Step 1: Write `src/ossagent/app.py`**

```python
"""Modal application: schedules and entry points."""
from __future__ import annotations
from pathlib import Path
import modal


app = modal.App("ossagent")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject.toml")
    .apt_install("git")
)

vol = modal.Volume.from_name("ossagent-data", create_if_missing=True)
secrets = [modal.Secret.from_name("ossagent-secrets")]

CONFIG_MOUNT_PATH = Path("/app/config")
DB_PATH = Path("/data/attempts.db")


@app.function(
    image=image, schedule=modal.Period(minutes=30),
    cpu=0.5, memory=512, timeout=300,
    volumes={"/data": vol}, secrets=secrets,
    mounts=[modal.Mount.from_local_dir("config", remote_path="/app/config")],
)
async def poll_repos() -> None:
    """Scheduled tick: poll all watched repos, triage, notify."""
    from ossagent.config import load_models_config, load_watched_repos_config
    from ossagent.db import Database
    from ossagent.github_client import GitHubClient
    from ossagent.models import get_llm
    from ossagent.scheduler import poll_one_repo
    from ossagent.telegram import TelegramBot
    from ossagent.triager import Triager
    import structlog

    log = structlog.get_logger()

    models = load_models_config(CONFIG_MOUNT_PATH / "models.yaml")
    repos = load_watched_repos_config(CONFIG_MOUNT_PATH / "watched_repos.yaml")

    db = Database(DB_PATH)
    db.init_schema()

    gh = GitHubClient()
    triager_llm = get_llm("triager", config=models)
    triager = Triager(llm=triager_llm)
    telegram = TelegramBot()

    for repo in repos:
        try:
            await poll_one_repo(repo, gh=gh, triager=triager,
                                telegram=telegram, db=db)
        except Exception as e:
            log.exception("poll_failed",
                          repo=f"{repo.owner}/{repo.name}", error=str(e))


@app.local_entrypoint()
def main() -> None:
    """Manual trigger for testing: `modal run src/ossagent/app.py`."""
    poll_repos.remote()
```

- [ ] **Step 2: Verify the module imports**

```bash
.venv/bin/python -c "import ossagent.app; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/ossagent/app.py
git commit -m "feat(app): Modal app with scheduled poll_repos function"
```

---

## Task 11: Pre-commit + ruff/mypy gate

**Files:**
- Create: `.pre-commit-config.yaml`

- [ ] **Step 1: Write `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.7.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.13.0
    hooks:
      - id: mypy
        additional_dependencies:
          - pydantic>=2.9
          - types-PyYAML
        args: [--config-file=pyproject.toml]
```

- [ ] **Step 2: Install hooks**

```bash
.venv/bin/pre-commit install
```

Expected: `pre-commit installed at .git/hooks/pre-commit`.

- [ ] **Step 3: Run hooks across the codebase**

```bash
.venv/bin/pre-commit run --all-files
```

Expected: all hooks pass. If any fail, fix the reported issues and re-run.

- [ ] **Step 4: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "chore: pre-commit with ruff + mypy gating"
```

---

## Task 12: Deploy smoke test

**Requires:** Modal account, Moonshot API key, GitHub PAT, Telegram bot token & user ID. Do not proceed until these are ready (parallel-work checklist).

- [ ] **Step 1: Create Modal secret from your local .env file**

```bash
modal secret create ossagent-secrets --from-dotenv ~/.config/ossagent.env
```

Expected: `Secret 'ossagent-secrets' created successfully.`

- [ ] **Step 2: Deploy the Modal app**

```bash
.venv/bin/modal deploy src/ossagent/app.py
```

Expected: log lines like `App created`, `Function poll_repos`. Note the dashboard URL.

- [ ] **Step 3: Trigger a manual run**

```bash
.venv/bin/modal run src/ossagent/app.py::poll_repos
```

Expected: function executes; logs show issue fetching and triage decisions. You receive at least one Telegram message if `langchain-ai/langchain` has recent fit issues. If none today, run still succeeds — check dashboard logs.

- [ ] **Step 4: Inspect logs in dashboard**

Open the dashboard URL → Functions → `poll_repos` → last run logs. Confirm structured logs `polling`, `triaged`, `polled`.

- [ ] **Step 5: Verify cost ledger**

```bash
.venv/bin/modal volume get ossagent-data /attempts.db /tmp/attempts.db
sqlite3 /tmp/attempts.db "SELECT role, sum(input_tokens), sum(output_tokens), sum(cost_usd) FROM cost_ledger GROUP BY role;"
```

Expected: a row for `triager` with non-zero tokens and a small cost (under $0.10).

---

## Task 13: Initial README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# ossagent

A multi-agent open-source contribution bot. *Phase 1: skeleton — polling and triage only.*

## Status

Phase 1 deployed: every 30 min the bot polls `langchain-ai/langchain` and classifies new issues with a cheap LLM. Issues classified as fit are sent to a Telegram chat for review.

Subsequent phases will add the LangGraph agent that drafts fixes for fit issues. See `docs/superpowers/specs/2026-05-16-mend-pilot-design.md` for the full design.

## Architecture (Phase 1)

```
GitHub Issues → poll_repos (Modal cron, 30m) → Triager (LLM) → Telegram
                                                  ↓
                                              SQLite (cost ledger, attempts)
```

## Running locally

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

## Deploying

```bash
modal secret create ossagent-secrets --from-dotenv ~/.config/ossagent.env
modal deploy src/ossagent/app.py
```

## Cost (Phase 1)

Triager runs on Kimi: ~$0.30 per 1M input tokens. At ~30 issues/day × 500 tokens each = $0.005/day = ~$0.15/mo.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: Phase 1 README"
```

---

## Phase 1.5 — Deferred testing pass

Tests skipped per user request — to add after Phase 1 runs end-to-end. When you're ready, the testing pass will add:

- `tests/conftest.py` with `tmp_sqlite` fixture
- Unit tests for each `src/ossagent/*.py` module mocking external services
- `pytest-asyncio` + `pytest-httpx` deps in `[dev]`
- CI on GitHub Actions to run `pytest` on every push

This is a separate plan; the Phase 1 system can be running while you write tests.

---

## Self-Review Notes

Coverage vs spec `2026-05-16-mend-pilot-design.md`:

- ✅ §3 High-level architecture — scheduler implemented; worker (LangGraph) deferred to Phase 2.
- ✅ §5 Two-stage triage — Stage 1 in this plan; Stage 2 in Phase 2.
- ✅ §7 Model abstraction — factory + YAML config + telemetry callback all built.
- ✅ §8 Modal deployment — basic app skeleton, secrets, volume, schedule.
- ❌ §4 The LangGraph — explicitly deferred to Phase 2 plan.
- ❌ §6 Telegram interaction (webhook) — outbound only this phase; webhook in Phase 2.
- ❌ §9 Evaluation harness — Phase 4 plan.

Phase 1 acceptance per spec: *"scheduler fires, lists new issues for one repo, sends each to Telegram as raw text. No agent yet."* — Met.

---

## Next plan

After Phase 1 runs cleanly for a few days, the next plan is `2026-05-16-phase-2-working-agent.md`: full LangGraph for all 4 classifications, `load_repo_context`, `enforce_style`, Critic, Telegram webhook for single-click approval, push + `gh pr create --draft`. Phase 3 (`2026-05-17-phase-3-eval-and-polish.md`) then runs the eval and ships the README with real numbers.
