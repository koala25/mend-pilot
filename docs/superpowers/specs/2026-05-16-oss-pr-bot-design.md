# OSS PR Bot — Design Document

**Date:** 2026-05-16
**Status:** Draft, awaiting user review
**Working name:** `ossagent` (final name TBD)

---

## 1. Overview

A LangGraph-orchestrated multi-agent pipeline that watches selected open-source repositories, attempts fixes for ripe issues, and drafts pull requests for human approval before submission. The system is deployed on Modal, controlled via a Telegram bot with single-click approval, and is model-agnostic by design (default: Kimi via Moonshot's OpenAI-compatible API; pluggable to Anthropic Claude or others via config).

### 1.1 Primary goals

1. **Portfolio-grade interview artifact** — a real, deployed system with merged PRs to LangChain / Tiangolo ecosystem repos and measurable solve rate on a published eval set.
2. **Multi-agent architecture with genuine technical depth** — LangGraph state machine with 7 specialized roles, conditional routing, retry loops, sandboxed test execution, cost-aware model routing, observability.
3. **Responsible AI engineering** — every PR passes through human approval before submission; OSS maintainers see honest disclosure, not slop.

### 1.2 Non-goals (v1)

- Multi-file or cross-repo refactors
- Embedding-based code retrieval (AST + grep is sufficient for our scope)
- Auto-merge or fully autonomous PR submission
- Custom test runners beyond `pytest`
- Cross-issue learning / memory between attempts
- Public web UI (Telegram + GitHub are the only surfaces)

### 1.3 Success criteria

- **Quantitative:** ≥10 PRs marked ready-for-review across the watched repos within 8 weeks of v1 ship; ≥5 of those eventually merged; published eval-set solve rate >25% on a 100-PR labeled set.
- **Qualitative:** README contains specific numbers (median $/attempt, solve rate, classification accuracy), a 60-second demo video, and an architecture diagram. Zero maintainer complaints about low-effort PRs.

---

## 2. Decisions made (recap of brainstorming)

| Decision | Value |
|---|---|
| Issue scope | Mixed: Triager routes each issue into one of {TYPO, DEPRECATION, TEST_GAP, BUG_FIX} based on confidence |
| Autonomy | Always draft mode — bot prepares fix, user single-click approves via Telegram, then bot pushes & opens GitHub draft PR |
| Initial target repos | LangChain + LangGraph cluster; FastAPI + Pydantic + SQLModel (Tiangolo) cluster |
| Orchestration framework | LangGraph (state machine with conditional edges, checkpointing) |
| Default LLM | Kimi (Moonshot API, OpenAI-compatible) — model name pinned at implementation time |
| Model abstraction | Single YAML config maps each agent role to a provider/model spec; one-line swap to A/B providers |
| Deployment | Modal (serverless, scale-to-zero, scheduled cron + web webhook) |
| Notification & control channel | Telegram bot with inline keyboard buttons for approve/reject |
| Cost target | $15-25/mo recurring with hard kill switches; max $3/attempt, $50/mo cap |

---

## 3. High-level architecture

```
            ┌──────────────────────────────┐
            │  Modal scheduled cron (30m)  │
            │  → poll watched repos        │
            │  → cheap Triager (Kimi tier) │
            │  → enqueue ripe issues       │
            └──────────────┬───────────────┘
                           ↓
            ┌──────────────────────────────┐
            │  Modal worker function       │
            │  (containerized, per-issue)  │
            │                              │
            │  ┌────────────────────────┐  │
            │  │ LangGraph agent loop   │  │ ← Section 4
            │  │ (7 roles, state, retry)│  │
            │  └─────────┬──────────────┘  │
            │            ↓                 │
            │  prepare draft on /data      │
            │  (branch in local clone,     │
            │   not yet pushed to GitHub)  │
            └──────────────┬───────────────┘
                           ↓
            ┌──────────────────────────────┐
            │  Telegram bot message        │
            │  → diff summary              │
            │  → Critic verdict + conf %   │
            │  → [✅ Approve] [❌ Reject]   │
            └──────────────┬───────────────┘
                           ↓ user taps
            ┌──────────────────────────────┐
            │  Modal web endpoint          │
            │  (Telegram webhook handler)  │
            │  → push branch + gh pr create│
            │    --draft                   │
            └──────────────┬───────────────┘
                           ↓
            ┌──────────────────────────────┐
            │  GitHub draft PR opened      │
            │  User reviews on github.com  │
            │  → marks ready-for-review    │
            │  → maintainer merges (eval)  │
            └──────────────────────────────┘
```

### 3.1 Runtime model

- **Always-on?** No. Pure scale-to-zero. Modal spins up only on cron firings or webhook hits.
- **Polling interval:** 30 min default. Adjustable per-repo.
- **Per-issue isolation:** each attempt runs in its own Modal worker container with its own clone (cached on volume).
- **State persistence:** SQLite on a Modal volume holds attempt history, retry counts, cost ledger, prepared-draft index.

---

## 4. The LangGraph

### 4.0 Worker entry point and lifecycle

The LangGraph is not invoked directly by the scheduler. It is invoked by the Modal worker function (`process_issue`), which wraps it with **pre-processing guards** and **post-processing bookkeeping**. These live outside the graph because they don't need agent state, must fail fast, and should not consume LangGraph's recursion budget.

**Pre-processing (in order, all fail-fast):**

1. **Budget guards** — check daily/monthly cost ledger. Above 95% of monthly cap → alert Telegram & exit. Above daily cap → silent exit (resumes next day).
2. **Idempotency check** — look up the issue in `attempts.db`:
   - `drafted_awaiting_approval` → exit (user still has to tap)
   - `merged` → exit (already shipped)
   - `attempt_count >= MAX_ATTEMPTS_PER_ISSUE` (default 3) → exit (give up)
   - Last attempt < 2 days ago → exit (cooldown to avoid thrashing)
3. **Freshness check** — fetch issue from GitHub: if `closed` or `assignee is not None`, mark skipped and exit.
4. **Per-repo rate-limit** — ≤5 active attempts per repo per day. Prevents spam-flagging by maintainers.
5. **Attempt record** — create a new row in `attempts.db` with a fresh UUID (`attempt_id`).
6. **Initial state assembly** — populate `AgentState` with `issue_url`, `classification`, `issue`, `attempt_id`, zero-initialized counters.
7. **Graph compile (cached)** — `get_compiled_graph()` returns a module-level singleton. No per-attempt graph rebuild.
8. **Run config** — set `thread_id = attempt_id` for checkpointing, attach `CostTracker` and `LangSmithTracer` as callbacks, set a hard `recursion_limit = 50` as a final safety net.

**Invocation:**

```python
try:
    final_state = graph.invoke(initial_state, config=config)
except Exception as e:
    attempts_db.mark_failed(attempt_id, reason=str(e))
    log_telegram_error(attempt_id, e)
    return
```

**Post-processing:**

- If `final_state.pr_metadata` is set: persist as `drafted_awaiting_approval`. (The `send_tg` node already pushed the Telegram message; nothing else to do here.)
- Otherwise: persist as `skipped` with the reason (`budget_exceeded` / `critic_aborted` / `style_failure` / `cant_reproduce` / etc.).
- Update the cost ledger.

**Why guards live in the worker, not in the graph:**

- Cleaner: the LangGraph is purely about *how to fix the issue*, not *should we fix it*.
- Cheaper: skipping in pre-processing costs ~$0 (no LLM calls); skipping after several graph nodes costs real money.
- More observable: ledger and state-transition logs separate the "agent did something interesting" signal from the "we declined to try" signal.

### 4.1 Node graph

```
load_issue → clone_repo → load_repo_context → plan → locate
                                                       │
                                           ┌───────────┴───────────┐
                                           │   is BUG_FIX?         │
                                           └───────────┬───────────┘
                                                   yes ↓ no
                                              reproduce │
                                                       ┌┴─→ implement → enforce_style
                                                                              │
                                                                ┌─────────────┴─────────────┐
                                                                │  style ok?                │
                                                                └─────────────┬─────────────┘
                                                                          yes ↓ no
                                                                              │ → implement (retry, same retry counter)
                                                                              ↓
                                                                  ┌──────────┴──────────┐
                                                                  │  is TEST_GAP?       │
                                                                  └──────────┬──────────┘
                                                                         yes ↓ no
                                                                     add_test │
                                                                             ┌┴─→ run_tests → critic → decide_next
                                                                                                          │
                                                                       ┌──────────────────────────────────┤
                                                                  PASS ↓                  RETRY  ↓        ↓ ABORT
                                                             prepare_pr             → plan (loop)         log_skip
                                                                  ↓                  (if retries < 2)         ↓
                                                              send_tg                                       END
                                                                  ↓
                                                                 END
```

### 4.2 State schema

```python
class AgentState(TypedDict):
    # Input
    issue_url: str
    repo_url: str
    classification: Literal["TYPO", "DEPRECATION", "TEST_GAP", "BUG_FIX"]

    # Loaded
    issue: IssueData
    repo_path: Path
    repo_context: RepoContext        # see 4.7

    # Planning / locating
    plan: list[PlanStep]
    target_files: list[TargetFile]   # (path, line_range, why)

    # Style enforcement
    style_violations: list[StyleViolation]   # empty after enforce_style passes
    style_retry_count: int                   # inner loop, separate from agent-level retries

    # Reproduction (BUG_FIX only)
    failing_test_path: Path | None
    failing_test_output: str | None

    # Implementation
    patch: str                       # unified diff

    # Validation
    test_run: TestRunResult
    critic_verdict: Literal["PASS", "RETRY", "ABORT"]
    critic_reasoning: str
    confidence: float                # 0.0–1.0

    # Bookkeeping
    retry_count: int
    cost_so_far: float
    elapsed_seconds: float
    attempt_id: str                  # UUID for Telegram callback_data

    # Output
    pr_metadata: PRMetadata | None   # title, body, branch
```

### 4.3 Nodes

| Node | Type | Model role | Behavior |
|---|---|---|---|
| `load_issue` | I/O | none | Fetch via GitHub REST API; cache to volume. |
| `clone_repo` | I/O | none | Shallow clone (`--depth 1`); cached across runs. |
| `load_repo_context` | I/O + light parse | none | Read & cache per-repo conventions: `CONTRIBUTING.md`, `README.md`, `.github/PULL_REQUEST_TEMPLATE.md`, `pyproject.toml`, `.pre-commit-config.yaml`, plus repo-specific docs (e.g. LangChain's `docs/contributing/code/code.mdx`). Cache 7 days or until upstream commit SHA changes. |
| `plan` | LLM | `planner` | Numbered plan; revised on retry using Critic feedback. Prompt includes excerpts from `CONTRIBUTING.md`. |
| `locate` | LLM + tools | `locator` | Tools: `grep`, `read_file`, `list_dir`, `tree_sitter_query`. AST-grounded file selection only. |
| `reproduce` | LLM + sandbox | `implementer` | (BUG_FIX) Write failing test, run, confirm fail. If can't repro → ABORT. |
| `implement` | LLM + tools | `implementer` | Tools: `read_file`, `apply_diff`. Output unified diff only — never full files. Prompt includes style conventions extracted from `repo_context`. |
| `enforce_style` | Pure tools | none | Run repo's own linters/formatters on the diff: `ruff check --fix`, `ruff format`, `mypy`, plus any `.pre-commit-config.yaml` hooks. Violations → loop back to `implement` (up to 2 style retries, separate from agent-level retries). Persistent failure → ABORT. |
| `add_test` | LLM | `tester` | (TEST_GAP) Write tests for targeted public symbols. Uses test patterns from `repo_context`. |
| `run_tests` | I/O | none | `pytest --timeout=120` via `subprocess.run`. 5 min wall-clock cap. |
| `critic` | LLM | `critic` | Reads issue + diff + test output + `CONTRIBUTING.md` excerpt. Outputs `PASS / RETRY / ABORT` + confidence. <0.6 confidence → forced RETRY/ABORT. |
| `decide_next` | logic | none | Routes on verdict + retry count + budget remaining. |
| `prepare_pr` | LLM | `pr_writer` | Fills the **actual** `PULL_REQUEST_TEMPLATE.md` (or repo's CONTRIBUTING-mandated PR format). Conventional-commit title. Issue link, summary, test plan, confidence, mandatory AI-disclosure footer. |
| `send_tg` | I/O | none | Build inline keyboard, POST to Telegram Bot API. |
| `log_skip` | I/O | none | If confidence threshold not met: optionally post courteous comment on the issue. |

### 4.4 Conditional routing

```python
def route_after_locate(state):
    return "reproduce" if state["classification"] == "BUG_FIX" else "implement"

def route_after_enforce_style(state):
    if not state["style_violations"]:
        return "add_test" if state["classification"] == "TEST_GAP" else "run_tests"
    if state["style_retry_count"] < MAX_STYLE_RETRIES:
        return "implement"           # tight inner loop, no cost to plan
    return "log_skip"                # give up if linters persistently fail

def decide_next(state):
    if state["cost_so_far"] > MAX_ATTEMPT_BUDGET:        return "log_skip"
    if state["critic_verdict"] == "ABORT":               return "log_skip"
    if state["critic_verdict"] == "PASS":                return "prepare_pr"
    if state["critic_verdict"] == "RETRY" and state["retry_count"] < MAX_RETRIES:
                                                          return "plan"
    return "log_skip"
```

### 4.5 Retry semantics

Two retry layers, intentionally separated:

| Layer | Trigger | Loops back to | Cap | Reasoning |
|---|---|---|---|---|
| Style (inner) | Linter / formatter / pre-commit failed | `implement` | `MAX_STYLE_RETRIES = 2` | Cheap, mechanical; replanning is overkill |
| Agent (outer) | Critic verdict RETRY | `plan` | `MAX_RETRIES = 2` | Critic feedback may invalidate the plan itself |

Hard caps:
- `MAX_ATTEMPT_BUDGET = $3.00` (kill switch).
- `MAX_WALL_CLOCK = 10 min` (Modal function timeout).

### 4.6 Checkpointing

`SqliteSaver` pointed at a SQLite file on the Modal volume. Every node transition is persisted; mid-graph crashes resume from last checkpoint without re-spending tokens.

```python
checkpointer = SqliteSaver.from_conn_string("/data/checkpoints.db")
graph = builder.compile(checkpointer=checkpointer)
graph.invoke(initial_state, config={"configurable": {"thread_id": attempt_id}})
```

### 4.7 RepoContext — project conventions cache

`RepoContext` is a structured snapshot of repo-specific norms. Built once per repo, cached on the Modal volume for 7 days or until the repo's `HEAD` SHA advances by more than N commits (configurable). Loaded as a sidecar to every attempt.

```python
@dataclass
class RepoContext:
    # Raw markdown excerpts (LLM-readable)
    contributing_md: str | None        # CONTRIBUTING.md, truncated to ~6k tokens
    readme_summary: str                # LLM-summarized README, ~500 tokens
    pr_template: str | None            # .github/PULL_REQUEST_TEMPLATE.md verbatim

    # Parsed conventions (extracted by an LLM at cache-build time, cheap)
    style_notes: list[str]             # e.g., "Use Google-style docstrings", "Type hints required on all public APIs"
    test_patterns: list[str]           # e.g., "pytest fixtures live in conftest.py at module root"
    pr_norms: list[str]                # e.g., "Link the issue in PR title", "Add changelog entry"

    # Deterministic tool config (parsed from files)
    ruff_config: dict | None           # from pyproject.toml [tool.ruff]
    black_config: dict | None
    mypy_config: dict | None
    pre_commit_hooks: list[str]        # from .pre-commit-config.yaml
    test_command: str                  # default "pytest", overridden if repo specifies

    # Sample test file (read at cache-build, used as one-shot example)
    sample_test_path: Path
    sample_test_excerpt: str           # ~50 lines from a typical test file
```

**Build process (one-time per repo per cache window):**

1. Read the raw files: `CONTRIBUTING.md`, `README.md`, `.github/PULL_REQUEST_TEMPLATE.md`, `pyproject.toml`, `.pre-commit-config.yaml`.
2. Parse deterministic config (TOML/YAML) — no LLM.
3. One Kimi call (~$0.02) summarizes README and extracts conventions from CONTRIBUTING into `style_notes` / `test_patterns` / `pr_norms`.
4. Pick a representative test file (heuristic: largest test file under `tests/` not named `conftest.py`); store excerpt.
5. Persist as JSON + raw markdown to `/data/repo_context/<owner>/<repo>/<sha-prefix>.json`.

**How agents use it:**

- `plan` prompt: prepends `style_notes` + `pr_norms` so the plan respects conventions from step one.
- `implement` prompt: includes `style_notes` + `sample_test_excerpt` (for similarity).
- `enforce_style` runs the **actual tools** with `ruff_config` / `black_config` / `mypy_config` — no prompting required.
- `add_test` prompt: includes `test_patterns` + `sample_test_excerpt`.
- `critic` prompt: includes `style_notes` + `pr_norms` to assess conformance.
- `pr_writer`: fills `pr_template` directly. If no template, builds one from `pr_norms`.

**When to rebuild:** if HEAD has moved by >50 commits since cache, OR `CONTRIBUTING.md` / `pyproject.toml` / `.pre-commit-config.yaml` SHA has changed. Cheap to check (just `git log --oneline` on cached refs).

---

## 5. Two-stage triage

**Critical for cost.** The cheap Triager runs in the scheduler — not the worker — so unfit issues never spin up the heavy container.

### 5.1 Stage 1 (scheduler, cheap)

```python
@app.function(schedule=Period(minutes=30), cpu=0.5)
def poll_repos():
    for repo in WATCHED_REPOS:
        for issue in fetch_new_issues(repo, since=last_seen[repo]):
            verdict = quick_triager(issue)   # ~$0.002/call with Kimi cheap tier
            if verdict["fit"] and verdict["confidence"] > 0.7:
                process_issue.spawn(issue.url, classification=verdict["class"])
```

Triager prompt is strict, JSON-only:

```json
{
  "fit": true,
  "confidence": 0.85,
  "class": "DEPRECATION",
  "reason": "Issue body cites PydanticDeprecatedSince20 warning, single file affected, mechanical fix"
}
```

Rejects: feature requests, multi-file refactors, design discussions, vague reports without repro.

### 5.2 Stage 2 (worker, expensive)

Full LangGraph runs only on the issues Stage 1 approves. Worker container has the repo image, tooling, sandbox.

---

## 6. Telegram interaction

### 6.1 Outbound message

When a worker finishes preparing a draft, it sends to Telegram:

```
🤖 Draft ready for issue #1234 (langchain-ai/langchain)

Classification: DEPRECATION
Confidence: 0.84
Cost: $1.12
Retries: 0

Files changed: libs/core/langchain_core/runnables/base.py (+8 -3)

Critic verdict: PASS
"Replaces deprecated `Runnable.run()` with `.invoke()`; covered by
existing test in test_runnables.py::test_invoke_basic which I re-ran."

[ View diff ]  [ ✅ Approve & Open Draft PR ]  [ ❌ Reject ]
```

### 6.2 Callback handling

Telegram POSTs the button tap to a Modal web endpoint:

```python
@app.web_endpoint(method="POST", label="telegram-webhook")
def telegram_webhook(payload: dict):
    callback = payload.get("callback_query")
    if not callback: return {"ok": True}
    attempt_id, action = callback["data"].split(":")
    match action:
        case "approve": push_and_open_pr.spawn(attempt_id)
        case "reject":  cleanup_attempt.spawn(attempt_id)
        case "diff":    send_full_diff.spawn(attempt_id, callback["from"]["id"])
    return {"ok": True}
```

### 6.3 State across the gap

Between draft preparation and user tap, the prepared work lives:
- Git branch in the cloned repo on Modal volume
- Attempt record in SQLite with all metadata
- Diff in a small file for fast `View diff` retrieval

Auto-cleanup: any unapproved draft >7 days old is purged by a daily cron.

---

## 7. Model abstraction

### 7.1 Config

```yaml
# config/models.yaml
defaults:
  provider: moonshot
  api_base: https://api.moonshot.cn/v1
  model: kimi-latest   # PIN at implementation time
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

# To A/B with Claude on implementer only:
# roles:
#   implementer:
#     provider: anthropic
#     model: claude-opus-4-7
#     temperature: 0.2
#     max_tokens: 4000
```

### 7.2 Factory

```python
def get_llm(role: AgentRole) -> BaseChatModel:
    cfg = resolve_config(role)
    match cfg.provider:
        case "moonshot" | "openai":
            return ChatOpenAI(base_url=cfg.api_base, api_key=cfg.api_key,
                              model=cfg.model, temperature=cfg.temperature,
                              max_tokens=cfg.max_tokens)
        case "anthropic":
            return ChatAnthropic(model=cfg.model, temperature=cfg.temperature,
                                 max_tokens=cfg.max_tokens)
```

### 7.3 Telemetry wrapper

Every LLM call goes through a wrapper that logs: timestamp, role, model, input_tokens, output_tokens, estimated_cost_usd, latency_ms, attempt_id. Stored in SQLite. Used for the README cost analysis.

---

## 8. Modal deployment

### 8.1 App structure

```python
import modal

app = modal.App("ossagent")

python_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject.toml")
    .apt_install("git")
)

vol = modal.Volume.from_name("ossagent-data", create_if_missing=True)
secrets = [modal.Secret.from_name("ossagent-secrets")]

@app.function(schedule=modal.Period(minutes=30), cpu=0.5, memory=512,
              volumes={"/data": vol}, secrets=secrets)
def poll_repos(): ...

@app.function(image=python_image, cpu=2, memory=4096, timeout=600,
              volumes={"/data": vol}, secrets=secrets)
def process_issue(issue_url: str, classification: str): ...

@app.web_endpoint(method="POST", label="telegram-webhook")
def telegram_webhook(payload: dict): ...

@app.function(image=python_image, volumes={"/data": vol}, secrets=secrets)
def push_and_open_pr(attempt_id: str): ...

@app.function(schedule=modal.Period(days=1), volumes={"/data": vol})
def cleanup_stale_drafts(): ...
```

### 8.2 Secrets

- `MOONSHOT_API_KEY`
- `ANTHROPIC_API_KEY` (for future A/B)
- `GITHUB_TOKEN` (fine-grained PAT, scoped to public_repo + read:issues on watched repos only)
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_USER_ID` (your user ID — webhook ignores callbacks from anyone else)
- `LANGSMITH_API_KEY` (free tier)

### 8.3 Volume layout

```
/data
├── repos/                  # cached shallow clones (rsync-style refresh per attempt)
│   ├── langchain-ai/langchain/
│   └── tiangolo/fastapi/
├── checkpoints.db          # LangGraph SqliteSaver
├── attempts.db             # attempt history, cost ledger, prepared-draft index
├── drafts/<attempt_id>/    # prepared diff + metadata pre-approval
└── cache/                  # prompt-cache scratch (per-provider)
```

### 8.4 Kill switches

- Per-attempt budget: $3 hard cap. Worker aborts and logs.
- Daily budget: $5. Scheduler halts new attempts when breached.
- Monthly budget: $50. Cron auto-pauses at 95% of cap; Telegram alert sent.
- Rate limits: max 5 issues per repo per day to avoid spam-flagging.

---

## 9. Evaluation harness

### 9.1 Eval set construction

Goal: 100 labeled examples, ~25 per lane.

For each watched repo:
1. Scrape recently-merged PRs (last 12 months) that close exactly one issue.
2. Filter to single-file changes with <100 lines of diff.
3. Manually classify each into TYPO / DEPRECATION / TEST_GAP / BUG_FIX.
4. Store as: `{issue_url, true_classification, original_pr_url, original_diff, target_files}`.

This eval set is also publishable as part of the README — it has independent value.

### 9.2 Offline eval pipeline

```python
def eval_run(eval_set: list[Example]) -> EvalReport:
    for example in eval_set:
        result = ossagent.process_issue(example.issue_url, dry_run=True)
        compare(result.patch, example.original_diff)
        compare(result.classification, example.true_classification)
    return aggregate()
```

Metrics:
- **Solve rate** — does our patch make the relevant tests pass? (gold standard)
- **Classification accuracy** — Triager's correctness vs hand labels
- **Patch overlap** — Jaccard / ROUGE-style match vs original PR diff (secondary)
- **$/attempt** — median, p95
- **Latency** — median, p95 wall-clock
- **Critic calibration** — does confidence correlate with actual success?

### 9.3 Baseline

Single-shot LLM (one Implementer prompt with full issue + repo summary, no agent loop, no Critic, no sandbox). Same eval set. Lets us claim *"specialization improves solve rate by N% over baseline."* This is the headline number for the README.

### 9.4 Live metrics

Post-deployment, tracked continuously:
- % of bot drafts you approve via Telegram
- % of approved drafts that maintainers merge
- $/PR-merged
- Median latency from issue-creation to bot-draft-ready

---

## 10. Cost analysis

| Bucket | Estimate /mo | Notes |
|---|---|---|
| LLM (Kimi default, ~50 attempts × ~$0.30 avg) | $15 | Cheaper than Claude; weaker prompt caching factored in |
| LLM bursts (debug runs, eval reruns) | $5-10 | One-off, mostly during dev |
| Modal compute | $0-10 | $30 free credit covers most months |
| Telegram, GitHub, LangSmith free tier | $0 | |
| **Realistic total** | **$20-35** | |
| **Hard cap** | **$50** | Auto-pause at 95% |

If A/B switch to Claude Opus for implementer only: add $20-30/mo.

---

## 11. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| OSS maintainer rejection / "AI slop" complaint | Medium | Mandatory human review gate + AI disclosure footer + narrow initial scope + AI-friendly target repos |
| Runaway cost | Medium | Multi-tier kill switches, scheduler-level triage, retry cap, Modal free tier |
| Low solve rate makes project unimpressive | Medium | Headline metric is improvement-over-baseline, not absolute; eval set is itself a contribution |
| Moonshot/Kimi API instability | Low-Medium | Generic abstraction allows instant swap to Claude/OpenAI; retry-with-backoff |
| Test environment mismatch (repo deps missing) | Medium | Pin watched repos to ones with `pip install -e .[dev]` working out-of-box; per-repo Modal images if needed (Phase 3) |
| Stale drafts pile up unreviewed | Low | Daily cleanup cron purges >7-day drafts |
| Telegram webhook abused | Very Low | Validate `from.id == TELEGRAM_USER_ID` before processing |

---

## 12. Roadmap

### Phase 1 — Skeleton (Weeks 1-2)
- Modal app skeleton (scheduler, worker, webhook, volume, secrets)
- Model factory + config + telemetry wrapper
- GitHub auth + repo clone + issue fetch
- Telegram bot wired (outbound hello-world)
- **Acceptance:** scheduler fires, lists new issues for one repo, sends each to Telegram as raw text. No agent yet.

### Phase 2 — Single lane: DEPRECATION (Weeks 3-4)
- Implement the full LangGraph for DEPRECATION-only classification
- `load_repo_context` + cache layer (CONTRIBUTING, PR template, lint configs)
- Sandboxed test runs via subprocess
- `enforce_style` step running repo's own `ruff` / `black` / `mypy` / pre-commit
- Critic with refusal threshold
- Telegram approval flow → push + draft PR
- **Acceptance:** 5 deprecation drafts prepared on real LangChain issues; `ruff`/`black`/`mypy` clean on all of them; you approve 2 and they get opened as draft PRs.

### Phase 3 — Multi-lane (Weeks 5-6)
- Add TYPO, TEST_GAP, BUG_FIX lanes with their conditional nodes
- Reproducer node (BUG_FIX)
- Per-classification prompt tuning
- **Acceptance:** All 4 classifications attempted in production; ≥10 total drafts prepared.

### Phase 4 — Eval harness (Weeks 7-8)
- Build the 100-example labeled eval set
- Offline eval runner
- Establish single-shot baseline
- Publish numbers in README
- **Acceptance:** README has real solve-rate numbers and improvement-over-baseline.

### Phase 5 — Polish (Weeks 9-10)
- Demo video (60-sec Loom)
- Architecture diagram (one PNG)
- Cost dashboard / report
- Blog post writing up methodology
- Optional: A/B with Claude Opus for headline comparison
- **Acceptance:** Project README is recruiter-grade; you've used the system on a real channel of issues for ≥4 weeks.

---

## 13. Open questions / TBDs

These should be resolved before or during implementation:

1. **Pin exact Moonshot/Kimi model identifier** — verify from [platform.moonshot.cn](https://platform.moonshot.cn) docs. Current placeholder: `kimi-latest`.
2. **Confirm Moonshot prompt-caching mechanics** — does it cache by hash of system prompt prefix? What's the TTL? Affects cost math.
3. **Initial repo set sizing** — start with LangChain only for Phase 2, or both clusters from day one? Recommendation: LangChain only in Phase 2, add Tiangolo in Phase 3.
4. **Critic confidence threshold** — currently 0.6. Will need tuning against real outputs. Set up to be configurable.
5. **Disclosure footer wording** — exact phrasing of the AI-assisted PR disclosure. Draft: *"This PR was drafted by an automated agent ([repo link]). I reviewed the diff and tested it locally before submitting. Feedback on the agent's quality is welcomed."*
6. **Final project name** — `ossagent` is a placeholder. Rename before public push.
7. **Eval set publishing** — public dataset repo, or just the README? Recommendation: separate `oss-pr-bot-evals` repo, MIT-licensed.

---

## 14. What this project signals to recruiters

The interview opener:
> *"I built a multi-agent system that contributes to open source. Seven LangGraph-orchestrated specialist agents, deployed serverless on Modal, model-agnostic by design — default Kimi with one-line swap to Claude. Single-click approval via Telegram before any PR opens. Published eval set of 100 labeled bug-fix commits across LangChain and FastAPI; my agent improves solve rate by N% over single-shot baseline. 14 draft PRs prepared so far, 8 merged after my review. ~$22/month all-in."*

Concrete artifacts a recruiter can click:
- GitHub repo with code + README + architecture diagram + cost dashboard
- Public eval set repo
- 60-second demo video at the top of the README
- LangSmith trace dashboard (or screenshots)
- The merged PRs themselves on famous repos

---

*End of design document.*
