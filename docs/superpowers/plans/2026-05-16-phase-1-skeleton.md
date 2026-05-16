# Phase 1 — Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy a Modal app that polls watched OSS repos every 30 minutes, classifies new issues with a cheap LLM triager, persists results to SQLite, and sends Telegram notifications for fit issues. No agent execution yet — Phase 1 is plumbing only.

**Architecture:** A single Modal app with one scheduled function. Inside the worker: a model-agnostic LLM factory (Kimi default, Anthropic-pluggable), a thin GitHub API client (`httpx`), an outbound-only Telegram bot wrapper, a cost telemetry SQLite store, and a deterministic Triager that returns structured JSON.

**Tech Stack:** Python 3.12 · uv · LangChain (`langchain-openai` for Moonshot compatibility) · Modal · httpx · SQLite (stdlib) · pytest · ruff · mypy · pre-commit

---

## File Structure

```
oss-pr-bot/
├── pyproject.toml                 # deps, ruff/mypy/pytest config
├── .pre-commit-config.yaml        # ruff + mypy gating commits
├── .gitignore
├── config/
│   ├── models.yaml                # role → provider/model mapping
│   └── watched_repos.yaml         # repos to poll + their default branches
├── src/
│   └── ossagent/
│       ├── __init__.py
│       ├── config.py              # load YAML configs once
│       ├── models.py              # get_llm(role) factory
│       ├── telemetry.py           # CostTracker callback for LangChain
│       ├── db.py                  # SQLite schema + helpers
│       ├── github_client.py       # fetch_new_issues, fetch_issue
│       ├── telegram.py            # send_message (outbound only this phase)
│       ├── triager.py             # classify_issue → TriageVerdict
│       ├── scheduler.py           # poll_repos: orchestrates one tick
│       └── app.py                 # Modal app: @app.function(schedule=...)
└── tests/
    ├── test_config.py
    ├── test_models.py
    ├── test_telemetry.py
    ├── test_db.py
    ├── test_github_client.py
    ├── test_telegram.py
    ├── test_triager.py
    └── test_scheduler.py
```

**One file = one responsibility.** Modal app code (`app.py`) stays thin — just wires functions to schedules; logic lives in `scheduler.py` so it's testable without Modal.

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/ossagent/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create directory structure**

```bash
cd /Users/kbtg/codebase/personal/oss-pr-bot
mkdir -p src/ossagent tests config
touch src/ossagent/__init__.py tests/__init__.py
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
    "langchain-openai>=0.2",       # Moonshot OpenAI-compatible
    "langchain-anthropic>=0.3",    # future swap
    "httpx>=0.27",
    "pydantic>=2.9",
    "pyyaml>=6.0",
    "structlog>=24.4",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "pytest-httpx>=0.32",
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
ignore = ["E501"]  # handled by formatter

[tool.mypy]
python_version = "3.12"
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
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
htmlcov/
.DS_Store
```

- [ ] **Step 4: Write `tests/conftest.py`**

```python
"""Shared pytest fixtures."""
import pytest


@pytest.fixture
def tmp_sqlite(tmp_path):
    """Path to a fresh SQLite DB for each test."""
    return tmp_path / "test.db"
```

- [ ] **Step 5: Install deps with `uv`**

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

Expected: venv created at `.venv/`, all deps installed without errors.

- [ ] **Step 6: Verify ruff + pytest discover the project**

```bash
.venv/bin/ruff check src tests
.venv/bin/pytest --collect-only
```

Expected: ruff reports no issues; pytest collects 0 tests (none written yet) without errors.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore src/ tests/
git commit -m "chore: scaffold pyproject + dirs + tooling config"
```

---

## Task 2: YAML config loading

**Files:**
- Create: `config/models.yaml`
- Create: `config/watched_repos.yaml`
- Create: `src/ossagent/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path
import textwrap
import pytest
from ossagent.config import load_models_config, load_watched_repos_config, ModelSpec, WatchedRepo


def test_load_models_config_parses_role_overrides(tmp_path: Path) -> None:
    cfg_file = tmp_path / "models.yaml"
    cfg_file.write_text(textwrap.dedent("""
        defaults:
          provider: moonshot
          api_base: https://api.moonshot.cn/v1
          model: kimi-latest
          temperature: 0.1
          max_tokens: 2000
        roles:
          triager:
            temperature: 0.0
            max_tokens: 200
          implementer:
            provider: anthropic
            model: claude-opus-4-7
            max_tokens: 4000
    """))
    cfg = load_models_config(cfg_file)
    assert cfg["triager"] == ModelSpec(
        provider="moonshot", api_base="https://api.moonshot.cn/v1",
        model="kimi-latest", temperature=0.0, max_tokens=200,
    )
    assert cfg["implementer"] == ModelSpec(
        provider="anthropic", api_base=None,
        model="claude-opus-4-7", temperature=0.1, max_tokens=4000,
    )


def test_load_watched_repos_config(tmp_path: Path) -> None:
    cfg_file = tmp_path / "watched_repos.yaml"
    cfg_file.write_text(textwrap.dedent("""
        repos:
          - owner: langchain-ai
            name: langchain
            default_branch: master
          - owner: tiangolo
            name: fastapi
            default_branch: master
    """))
    repos = load_watched_repos_config(cfg_file)
    assert len(repos) == 2
    assert repos[0] == WatchedRepo(owner="langchain-ai", name="langchain", default_branch="master")


def test_load_models_raises_on_missing_role(tmp_path: Path) -> None:
    cfg_file = tmp_path / "models.yaml"
    cfg_file.write_text("defaults:\n  provider: moonshot\n  model: x\nroles: {}\n")
    with pytest.raises(KeyError, match="triager"):
        load_models_config(cfg_file)
```

- [ ] **Step 2: Run test to verify failure**

```bash
.venv/bin/pytest tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ossagent.config'`.

- [ ] **Step 3: Write `src/ossagent/config.py`**

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

- [ ] **Step 4: Run test to verify pass**

```bash
.venv/bin/pytest tests/test_config.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Write the production `config/models.yaml`**

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

- [ ] **Step 6: Write the production `config/watched_repos.yaml`**

```yaml
repos:
  - owner: langchain-ai
    name: langchain
    default_branch: master
```

(Phase 1 starts with one repo. Add others in later phases once stable.)

- [ ] **Step 7: Commit**

```bash
git add config/ src/ossagent/config.py tests/test_config.py
git commit -m "feat(config): YAML loaders for model and repo configs"
```

---

## Task 3: Model factory

**Files:**
- Create: `src/ossagent/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
from pathlib import Path
from unittest.mock import patch
import pytest
from ossagent.models import get_llm
from ossagent.config import ModelSpec


@pytest.fixture
def fake_config() -> dict[str, ModelSpec]:
    return {
        "triager": ModelSpec(
            provider="moonshot", api_base="https://api.moonshot.cn/v1",
            model="kimi-latest", temperature=0.0, max_tokens=200,
        ),
        "implementer": ModelSpec(
            provider="anthropic", api_base=None,
            model="claude-opus-4-7", temperature=0.2, max_tokens=4000,
        ),
    }


def test_get_llm_moonshot_uses_chat_openai_with_base_url(fake_config, monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test")
    with patch("ossagent.models.ChatOpenAI") as MockOpenAI:
        get_llm("triager", config=fake_config)
        MockOpenAI.assert_called_once()
        call_kwargs = MockOpenAI.call_args.kwargs
        assert call_kwargs["base_url"] == "https://api.moonshot.cn/v1"
        assert call_kwargs["model"] == "kimi-latest"
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["max_tokens"] == 200


def test_get_llm_anthropic(fake_config, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    with patch("ossagent.models.ChatAnthropic") as MockAnthropic:
        get_llm("implementer", config=fake_config)
        MockAnthropic.assert_called_once()
        call_kwargs = MockAnthropic.call_args.kwargs
        assert call_kwargs["model"] == "claude-opus-4-7"


def test_get_llm_raises_on_unknown_provider(fake_config):
    fake_config["triager"] = ModelSpec(
        provider="ollama", api_base=None, model="x",
        temperature=0.0, max_tokens=200,
    )
    with pytest.raises(ValueError, match="ollama"):
        get_llm("triager", config=fake_config)
```

- [ ] **Step 2: Run test to verify failure**

```bash
.venv/bin/pytest tests/test_models.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/ossagent/models.py`**

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

- [ ] **Step 4: Run test to verify pass**

```bash
.venv/bin/pytest tests/test_models.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/ossagent/models.py tests/test_models.py
git commit -m "feat(models): provider-agnostic LLM factory"
```

---

## Task 4: SQLite schema + helpers

**Files:**
- Create: `src/ossagent/db.py`
- Test: `tests/test_db.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db.py
from datetime import datetime, timedelta, UTC
from pathlib import Path
from ossagent.db import Database, Attempt, AttemptStatus, CostLedgerEntry


def test_init_creates_schema(tmp_sqlite: Path):
    db = Database(tmp_sqlite)
    db.init_schema()
    # Idempotent: running twice doesn't error
    db.init_schema()


def test_record_and_fetch_attempt(tmp_sqlite: Path):
    db = Database(tmp_sqlite)
    db.init_schema()
    attempt = Attempt(
        attempt_id="abc-123",
        issue_url="https://github.com/x/y/issues/1",
        repo_owner="x",
        repo_name="y",
        classification="DEPRECATION",
        status=AttemptStatus.IN_PROGRESS,
        started_at=datetime.now(UTC),
        attempt_count=1,
    )
    db.record_attempt(attempt)
    fetched = db.fetch_attempt_by_issue("https://github.com/x/y/issues/1")
    assert fetched is not None
    assert fetched.attempt_id == "abc-123"
    assert fetched.classification == "DEPRECATION"


def test_repo_attempts_today(tmp_sqlite: Path):
    db = Database(tmp_sqlite)
    db.init_schema()
    now = datetime.now(UTC)
    for i in range(3):
        db.record_attempt(Attempt(
            attempt_id=f"a-{i}", issue_url=f"https://github.com/x/y/issues/{i}",
            repo_owner="x", repo_name="y", classification="DEPRECATION",
            status=AttemptStatus.IN_PROGRESS, started_at=now, attempt_count=1,
        ))
    assert db.repo_attempts_today("x", "y") == 3


def test_cost_ledger_add_and_sum(tmp_sqlite: Path):
    db = Database(tmp_sqlite)
    db.init_schema()
    now = datetime.now(UTC)
    db.add_cost(CostLedgerEntry(attempt_id="a-1", role="triager",
                                input_tokens=100, output_tokens=20,
                                cost_usd=0.0005, at=now))
    db.add_cost(CostLedgerEntry(attempt_id="a-2", role="implementer",
                                input_tokens=2000, output_tokens=800,
                                cost_usd=1.20, at=now))
    assert abs(db.cost_today() - 1.2005) < 1e-6
    assert abs(db.cost_month_to_date() - 1.2005) < 1e-6


def test_last_seen_issue_per_repo(tmp_sqlite: Path):
    db = Database(tmp_sqlite)
    db.init_schema()
    db.set_last_seen_issue("x", "y", issue_number=42)
    assert db.get_last_seen_issue("x", "y") == 42
    assert db.get_last_seen_issue("x", "other") is None
```

- [ ] **Step 2: Run test to verify failure**

```bash
.venv/bin/pytest tests/test_db.py -v
```

Expected: FAIL with import error.

- [ ] **Step 3: Write `src/ossagent/db.py`**

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

- [ ] **Step 4: Run test to verify pass**

```bash
.venv/bin/pytest tests/test_db.py -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/ossagent/db.py tests/test_db.py
git commit -m "feat(db): SQLite schema for attempts, cost ledger, repo state"
```

---

## Task 5: Cost telemetry callback

**Files:**
- Create: `src/ossagent/telemetry.py`
- Test: `tests/test_telemetry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_telemetry.py
from pathlib import Path
from datetime import datetime, UTC
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.messages import AIMessage
from ossagent.telemetry import CostTracker, MODEL_PRICING
from ossagent.db import Database


def _result_with_usage(input_tokens: int, output_tokens: int, model: str) -> LLMResult:
    msg = AIMessage(
        content="hello",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        response_metadata={"model_name": model},
    )
    return LLMResult(
        generations=[[ChatGeneration(message=msg, text="hello")]],
        llm_output={"model_name": model},
    )


def test_cost_tracker_writes_ledger_entry(tmp_sqlite: Path):
    db = Database(tmp_sqlite)
    db.init_schema()
    tracker = CostTracker(db=db, attempt_id="a-1", role="triager",
                          model="kimi-latest")
    fake = _result_with_usage(1000, 100, "kimi-latest")
    tracker.on_llm_end(fake)
    assert db.cost_today() > 0


def test_model_pricing_known_models():
    assert "kimi-latest" in MODEL_PRICING
    assert MODEL_PRICING["kimi-latest"].input_per_1m > 0
    assert MODEL_PRICING["claude-opus-4-7"].input_per_1m > 0
```

- [ ] **Step 2: Run test to verify failure**

```bash
.venv/bin/pytest tests/test_telemetry.py -v
```

Expected: FAIL with import error.

- [ ] **Step 3: Write `src/ossagent/telemetry.py`**

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
    input_per_1m: float   # USD per 1M input tokens
    output_per_1m: float  # USD per 1M output tokens


# Prices as of 2026-05; pin from provider docs.
MODEL_PRICING: dict[str, ModelPrice] = {
    "kimi-latest":      ModelPrice(input_per_1m=0.30, output_per_1m=1.50),
    "claude-haiku-4-5": ModelPrice(input_per_1m=0.80, output_per_1m=4.00),
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
        if price is None:
            cost_usd = 0.0  # unknown model: log tokens but no $
        else:
            cost_usd = (input_tokens / 1_000_000) * price.input_per_1m \
                     + (output_tokens / 1_000_000) * price.output_per_1m
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

- [ ] **Step 4: Run test to verify pass**

```bash
.venv/bin/pytest tests/test_telemetry.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/ossagent/telemetry.py tests/test_telemetry.py
git commit -m "feat(telemetry): cost-tracking LangChain callback"
```

---

## Task 6: GitHub client

**Files:**
- Create: `src/ossagent/github_client.py`
- Test: `tests/test_github_client.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_github_client.py
import pytest
from httpx import Response
from ossagent.github_client import GitHubClient, Issue


@pytest.mark.asyncio
async def test_fetch_new_issues_filters_pull_requests(httpx_mock, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    httpx_mock.add_response(
        url="https://api.github.com/repos/x/y/issues?state=open&sort=created&direction=desc&per_page=30&since=2026-05-01T00%3A00%3A00",
        json=[
            {"number": 5, "title": "real issue", "body": "halp", "labels": [],
             "user": {"login": "alice"}, "html_url": "https://github.com/x/y/issues/5",
             "state": "open", "assignee": None, "comments": 2,
             "created_at": "2026-05-15T10:00:00Z"},
            {"number": 6, "title": "a PR", "body": "...", "labels": [],
             "user": {"login": "bob"}, "html_url": "https://github.com/x/y/pull/6",
             "state": "open", "assignee": None, "comments": 0,
             "pull_request": {"url": "..."},
             "created_at": "2026-05-15T11:00:00Z"},
        ],
    )
    client = GitHubClient()
    issues = await client.fetch_new_issues("x", "y", since="2026-05-01T00:00:00")
    assert len(issues) == 1
    assert issues[0].number == 5


@pytest.mark.asyncio
async def test_fetch_issue_returns_full_payload(httpx_mock, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    httpx_mock.add_response(
        url="https://api.github.com/repos/x/y/issues/42",
        json={"number": 42, "title": "deprecation warning",
              "body": "Found `Runnable.run` deprecated",
              "labels": [{"name": "good first issue"}],
              "user": {"login": "alice"},
              "html_url": "https://github.com/x/y/issues/42",
              "state": "open", "assignee": None, "comments": 0,
              "created_at": "2026-05-15T10:00:00Z"},
    )
    client = GitHubClient()
    issue = await client.fetch_issue("x", "y", 42)
    assert issue.title == "deprecation warning"
    assert "good first issue" in issue.labels
```

- [ ] **Step 2: Run test to verify failure**

```bash
.venv/bin/pytest tests/test_github_client.py -v
```

Expected: FAIL with import error.

- [ ] **Step 3: Write `src/ossagent/github_client.py`**

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
        return [
            self._to_issue(d) for d in data
            if "pull_request" not in d  # GitHub returns PRs through the same endpoint
        ]

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

- [ ] **Step 4: Run test to verify pass**

```bash
.venv/bin/pytest tests/test_github_client.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/ossagent/github_client.py tests/test_github_client.py
git commit -m "feat(github): async client for issue listing and fetching"
```

---

## Task 7: Telegram outbound bot

**Files:**
- Create: `src/ossagent/telegram.py`
- Test: `tests/test_telegram.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_telegram.py
import pytest
from ossagent.telegram import TelegramBot, TriageNotification


@pytest.mark.asyncio
async def test_send_triage_notification(httpx_mock, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "12345:abc")
    monkeypatch.setenv("TELEGRAM_USER_ID", "999")
    httpx_mock.add_response(
        url="https://api.telegram.org/bot12345:abc/sendMessage",
        method="POST",
        json={"ok": True, "result": {"message_id": 7}},
    )
    bot = TelegramBot()
    payload = TriageNotification(
        issue_url="https://github.com/x/y/issues/5",
        issue_title="Deprecation: foo()",
        classification="DEPRECATION",
        confidence=0.84,
        reason="single-file mechanical fix",
    )
    msg_id = await bot.send_triage_notification(payload)
    assert msg_id == 7
    sent = httpx_mock.get_requests()[0]
    body = sent.read().decode()
    assert "999" in body
    assert "DEPRECATION" in body
    assert "0.84" in body


@pytest.mark.asyncio
async def test_send_handles_api_failure(httpx_mock, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "12345:abc")
    monkeypatch.setenv("TELEGRAM_USER_ID", "999")
    httpx_mock.add_response(
        url="https://api.telegram.org/bot12345:abc/sendMessage",
        method="POST",
        status_code=400,
        json={"ok": False, "description": "Bad request"},
    )
    bot = TelegramBot()
    with pytest.raises(RuntimeError, match="Telegram"):
        await bot.send_triage_notification(TriageNotification(
            issue_url="x", issue_title="t", classification="DEPRECATION",
            confidence=0.5, reason="r",
        ))
```

- [ ] **Step 2: Run test to verify failure**

```bash
.venv/bin/pytest tests/test_telegram.py -v
```

Expected: FAIL with import error.

- [ ] **Step 3: Write `src/ossagent/telegram.py`**

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

- [ ] **Step 4: Run test to verify pass**

```bash
.venv/bin/pytest tests/test_telegram.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/ossagent/telegram.py tests/test_telegram.py
git commit -m "feat(telegram): outbound triage-notification bot"
```

---

## Task 8: Triager — Stage 1 LLM classifier

**Files:**
- Create: `src/ossagent/triager.py`
- Test: `tests/test_triager.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_triager.py
from unittest.mock import AsyncMock, MagicMock
import pytest
from ossagent.triager import Triager, TriageVerdict, Classification
from ossagent.github_client import Issue


@pytest.fixture
def deprecation_issue() -> Issue:
    return Issue(
        number=42, title="DeprecationWarning: Runnable.run()",
        body="Using `Runnable.run()` triggers PydanticDeprecatedSince20. Please update to `.invoke()`.",
        labels=["bug", "deprecation"], author="alice",
        html_url="https://github.com/x/y/issues/42", state="open",
        assignee=None, comments=2, created_at="2026-05-15T10:00:00Z",
    )


@pytest.fixture
def vague_issue() -> Issue:
    return Issue(
        number=7, title="App is slow",
        body="Sometimes my app is slow, please fix.",
        labels=[], author="bob", html_url="https://github.com/x/y/issues/7",
        state="open", assignee=None, comments=0, created_at="2026-05-15T10:00:00Z",
    )


def _mock_llm_returning(json_text: str) -> MagicMock:
    llm = MagicMock()
    msg = MagicMock()
    msg.content = json_text
    llm.ainvoke = AsyncMock(return_value=msg)
    return llm


@pytest.mark.asyncio
async def test_triager_classifies_deprecation_as_fit(deprecation_issue):
    llm = _mock_llm_returning(
        '{"fit": true, "confidence": 0.86, "class": "DEPRECATION", '
        '"reason": "single-file mechanical fix"}'
    )
    triager = Triager(llm=llm)
    verdict = await triager.classify(deprecation_issue)
    assert verdict.fit is True
    assert verdict.classification == Classification.DEPRECATION
    assert verdict.confidence > 0.8


@pytest.mark.asyncio
async def test_triager_rejects_vague_issue(vague_issue):
    llm = _mock_llm_returning(
        '{"fit": false, "confidence": 0.3, "class": "BUG_FIX", '
        '"reason": "no repro, too vague"}'
    )
    triager = Triager(llm=llm)
    verdict = await triager.classify(vague_issue)
    assert verdict.fit is False
    assert "vague" in verdict.reason.lower()


@pytest.mark.asyncio
async def test_triager_handles_malformed_json(deprecation_issue):
    llm = _mock_llm_returning("not json")
    triager = Triager(llm=llm)
    verdict = await triager.classify(deprecation_issue)
    assert verdict.fit is False
    assert verdict.confidence == 0.0
```

- [ ] **Step 2: Run test to verify failure**

```bash
.venv/bin/pytest tests/test_triager.py -v
```

Expected: FAIL with import error.

- [ ] **Step 3: Write `src/ossagent/triager.py`**

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

- [ ] **Step 4: Run test to verify pass**

```bash
.venv/bin/pytest tests/test_triager.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/ossagent/triager.py tests/test_triager.py
git commit -m "feat(triager): Stage 1 LLM classifier with strict JSON parsing"
```

---

## Task 9: Scheduler orchestration logic

**Files:**
- Create: `src/ossagent/scheduler.py`
- Test: `tests/test_scheduler.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scheduler.py
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest
from ossagent.scheduler import poll_one_repo
from ossagent.config import WatchedRepo
from ossagent.db import Database
from ossagent.github_client import Issue
from ossagent.triager import TriageVerdict, Classification


def _issue(n: int) -> Issue:
    return Issue(
        number=n, title=f"issue {n}", body="body", labels=[],
        author="x", html_url=f"https://github.com/o/r/issues/{n}",
        state="open", assignee=None, comments=0,
        created_at="2026-05-15T10:00:00Z",
    )


@pytest.mark.asyncio
async def test_poll_one_repo_sends_only_fit_issues(tmp_sqlite: Path):
    db = Database(tmp_sqlite)
    db.init_schema()
    repo = WatchedRepo(owner="o", name="r", default_branch="main")

    gh = MagicMock()
    gh.fetch_new_issues = AsyncMock(return_value=[_issue(1), _issue(2), _issue(3)])

    triager = MagicMock()
    triager.classify = AsyncMock(side_effect=[
        TriageVerdict(fit=True, confidence=0.9,
                      classification=Classification.DEPRECATION, reason="r1"),
        TriageVerdict(fit=False, confidence=0.3,
                      classification=Classification.BUG_FIX, reason="r2"),
        TriageVerdict(fit=True, confidence=0.8,
                      classification=Classification.TYPO, reason="r3"),
    ])

    tg = MagicMock()
    tg.send_triage_notification = AsyncMock(return_value=42)

    await poll_one_repo(repo, gh=gh, triager=triager, telegram=tg, db=db)

    # Only fit issues notified
    assert tg.send_triage_notification.call_count == 2

    # last_seen updated to max number seen
    assert db.get_last_seen_issue("o", "r") == 3


@pytest.mark.asyncio
async def test_poll_one_repo_respects_last_seen(tmp_sqlite: Path):
    db = Database(tmp_sqlite)
    db.init_schema()
    db.set_last_seen_issue("o", "r", 5)
    repo = WatchedRepo(owner="o", name="r", default_branch="main")

    gh = MagicMock()
    gh.fetch_new_issues = AsyncMock(return_value=[])  # nothing new

    triager = MagicMock()
    tg = MagicMock()

    await poll_one_repo(repo, gh=gh, triager=triager, telegram=tg, db=db)

    # Called with `since` derived from last_seen — we just check it was called
    gh.fetch_new_issues.assert_awaited_once()
    triager.classify.assert_not_called()
```

- [ ] **Step 2: Run test to verify failure**

```bash
.venv/bin/pytest tests/test_scheduler.py -v
```

Expected: FAIL with import error.

- [ ] **Step 3: Write `src/ossagent/scheduler.py`**

```python
"""Per-tick orchestration: poll a repo, triage new issues, notify."""
from __future__ import annotations
from datetime import datetime, timedelta, UTC
import structlog
from ossagent.config import WatchedRepo
from ossagent.db import Database
from ossagent.github_client import GitHubClient, Issue
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
    # First run: look back 24h; otherwise look back 1h to handle drift.
    delta = timedelta(hours=24) if last_seen is None else timedelta(hours=1)
    return (datetime.now(UTC) - delta).strftime("%Y-%m-%dT%H:%M:%S")
```

- [ ] **Step 4: Run test to verify pass**

```bash
.venv/bin/pytest tests/test_scheduler.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/ossagent/scheduler.py tests/test_scheduler.py
git commit -m "feat(scheduler): per-tick repo polling + triage + notify"
```

---

## Task 10: Modal app wiring

**Files:**
- Create: `src/ossagent/app.py`

This task has no unit test — Modal functions are integration-tested via a deploy. The next task is the deploy smoke test.

- [ ] **Step 1: Write `src/ossagent/app.py`**

```python
"""Modal application: schedules and entry points."""
from __future__ import annotations
import asyncio
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
    # mount the config/ directory at /app/config
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
            # Don't let one repo's failure stop the others.
            import structlog
            structlog.get_logger().exception("poll_failed",
                                            repo=f"{repo.owner}/{repo.name}",
                                            error=str(e))


@app.local_entrypoint()
def main() -> None:
    """Manual trigger for testing: `modal run src/ossagent/app.py`."""
    poll_repos.remote()
```

- [ ] **Step 2: Verify the file is importable locally**

```bash
.venv/bin/python -c "import ossagent.app; print('ok')"
```

Expected: prints `ok`. (Modal decorators are lazy — local import works without a Modal connection.)

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
        exclude: ^tests/
```

- [ ] **Step 2: Install hooks**

```bash
.venv/bin/pre-commit install
```

Expected: `pre-commit installed at .git/hooks/pre-commit`

- [ ] **Step 3: Run hooks across the codebase**

```bash
.venv/bin/pre-commit run --all-files
```

Expected: all hooks pass; if any fail, fix issues, re-run.

- [ ] **Step 4: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "chore: pre-commit with ruff + mypy gating"
```

---

## Task 12: Deploy smoke test

This task has manual steps that require your parallel-work credentials. **Do not proceed until all critical-path items from the parallel-work checklist are done.**

- [ ] **Step 1: Set Modal secret from your local .env file**

```bash
# Assumes you have created ~/.config/ossagent.env per the checklist
modal secret create ossagent-secrets --from-dotenv ~/.config/ossagent.env
```

Expected: `Secret 'ossagent-secrets' created successfully.`

- [ ] **Step 2: Deploy the Modal app**

```bash
.venv/bin/modal deploy src/ossagent/app.py
```

Expected: log lines like `App created` and `Function poll_repos`. Note the dashboard URL printed.

- [ ] **Step 3: Trigger a manual run (don't wait for the cron)**

```bash
.venv/bin/modal run src/ossagent/app.py::poll_repos
```

Expected: function executes, logs show issue fetching and triage decisions, you receive at least one Telegram message if there are recent fit issues in `langchain-ai/langchain`.

If no fit issues today: the run still succeeds, you just don't get a Telegram message. Check Modal dashboard logs to confirm the triager ran on candidate issues.

- [ ] **Step 4: Tail logs in dashboard**

Open the dashboard URL from Step 2 → Functions → `poll_repos` → check logs of the last run. Confirm structured logs `polling`, `triaged`, `polled`.

- [ ] **Step 5: Verify cost ledger after the run**

```bash
# Modal volume-stored DB; fetch it locally to inspect
.venv/bin/modal volume get ossagent-data /attempts.db /tmp/attempts.db
sqlite3 /tmp/attempts.db "SELECT role, sum(input_tokens), sum(output_tokens), sum(cost_usd) FROM cost_ledger GROUP BY role;"
```

Expected: a row for `triager` with non-zero tokens and a small cost (under $0.10 for a single run on one repo).

- [ ] **Step 6: Commit any docs added during smoke test**

If you wrote any notes/runbooks during the smoke test:

```bash
git add docs/
git commit -m "docs: phase 1 smoke-test runbook notes"
```

Otherwise skip this step.

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

Subsequent phases will add the LangGraph agent that drafts fixes for fit issues. See `docs/superpowers/specs/2026-05-16-oss-pr-bot-design.md` for the full design.

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
.venv/bin/pytest
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

## Self-Review Notes

Coverage vs the design spec (`2026-05-16-oss-pr-bot-design.md`):

- ✅ §3 High-level architecture — scheduler is implemented; worker (LangGraph) deferred to Phase 2.
- ✅ §5 Two-stage triage — Stage 1 (scheduler) is in this plan; Stage 2 (worker) is in Phase 2.
- ✅ §7 Model abstraction — factory + YAML config + telemetry callback all built.
- ✅ §8 Modal deployment — basic app skeleton, secrets, volume, schedule.
- ❌ §4 The LangGraph — explicitly deferred to Phase 2 (separate plan).
- ❌ §6 Telegram interaction (webhook) — outbound only this phase; webhook in Phase 2.
- ❌ §9 Evaluation harness — Phase 4 (separate plan).

Phase 1 produces working, testable software: a deployed Modal app that polls, classifies, and notifies. No agent, no PRs opened. Acceptance from the spec's Phase 1 row is met: *"scheduler fires, lists new issues for one repo, sends each to Telegram as raw text. No agent yet."*

---

## Next plan

After Phase 1 lands and you've verified Telegram notifications come through cleanly for ~3 days of real polling, the next plan to write is `2026-XX-XX-phase-2-deprecation-lane.md` — adding `load_repo_context`, the full LangGraph for DEPRECATION classification, `enforce_style`, the Critic, and the Telegram webhook for single-click PR approval.
