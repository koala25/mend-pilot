# Phase 2 Implementation — Handoff

**Date:** 2026-05-17
**Status:** Tasks 1–10 of 18 complete and committed. Task 11 not started.
**Working tree:** clean. 10 commits ahead of `origin/main`.

---

## How to resume

1. Read this file end-to-end.
2. Read the plan: `docs/superpowers/plans/2026-05-16-phase-2-working-agent.md`.
3. Pick up at **Task 11** (Critic + prepare_pr + send_tg + Telegram approval button).
4. Continue through Task 18.
5. **Do not amend or rewrite the 10 commits already on `main`.** Continue with new commits.

### Hard rules from the user

- **Author all commits as Kushal Bakliwal (`kushalbakliwal25@gmail.com`).** This is the personal account. Per-repo `git config` is already set correctly — verify with `git config user.email`.
- **No "Claude", "AI", "Anthropic", "Co-Authored-By", or assistant references anywhere in commit messages.** This overrides the default Claude Code commit-message template. Drop the trailer.
- The PR-writer system prompts in `nodes.py` (when added in Task 11) already strip AI tells from PR output — that is a separate constraint baked into the agent itself.

### Verify before touching anything

```bash
cd /Users/kbtg/codebase/personal/mend-pilot
git config user.email   # → kushalbakliwal25@gmail.com
git config user.name    # → Kushal Bakliwal
git log --oneline -12   # Top should be a8d2d28 feat(agent): enforce_style + run_tests nodes
git status              # → clean
.venv/bin/python -c "from ossagent.agent.nodes import make_enforce_style_node, make_run_tests_node; print('ok')"
```

---

## Completed (committed)

| # | Task | Commit | Notes |
|---|------|--------|-------|
| 1 | Phase 2 deps | `fa5f992` | Added langgraph, langgraph-checkpoint-sqlite, tree-sitter, tree-sitter-python, PyGithub. `uv.lock` regenerated and committed. |
| 2 | `AgentState` + dataclasses | `d257b0d` | `src/ossagent/agent/state.py`. **Deviation:** Used `TYPE_CHECKING` guard for `RepoContext` import (forward-ref string broke ruff/mypy under strict mode). |
| 3 | Helper utilities | `7d2512d` | `src/ossagent/agent/tools.py`. git, diff, ripgrep, file-window. |
| 4 | Tree-sitter AST locator | `e61faa9` | `src/ossagent/agent/ast_locator.py`. |
| 5 | `RepoContext` loader | `e731954` | `src/ossagent/agent/context.py`. **Deviation:** Made `load_repo_context` **async** (plan had it sync, but the extractor returned is `async def` and would never be awaited). Worker in Task 13 must `await` it. |
| 6 | Convention extractor (LLM) | `86cee3e` | `src/ossagent/agent/context_extractor.py`. |
| 7 | Plan + locate nodes | `11b39ad` | `src/ossagent/agent/nodes.py`. Added a `NodeFn` type alias so all `make_*_node` factories have explicit return types (mypy strict mode). |
| 8 | Reproduce node (BUG_FIX) | `0566018` | Appended to `nodes.py`. |
| 9 | Implement + add_test nodes | `10b7484` | Appended. |
| 10 | Enforce style + run tests | `a8d2d28` | Appended. |

All ten commits passed `pre-commit` (ruff, ruff-format, mypy-strict).

---

## Pending (not started)

| # | Task | Files | Key gotchas |
|---|------|-------|-------------|
| 11 | Critic + prepare_pr + send_tg | `nodes.py` (append), `src/ossagent/telegram.py` (add `send_draft_for_approval` method to `TelegramBot` class) | Two-pass humanizer prompt is in plan §11. **Do not modify the prompts** — they are intentional. Add `PRMetadata` to the `state.py` imports in `nodes.py`. |
| 12 | LangGraph wiring | `src/ossagent/agent/graph.py` | Plan uses `SqliteSaver.from_conn_string`. Newer langgraph-checkpoint-sqlite (3.x) may want `AsyncSqliteSaver` or a context manager. Verify with `.venv/bin/python -c "from langgraph.checkpoint.sqlite import SqliteSaver"` before writing the call. |
| 13 | Worker (`process_issue`) | `src/ossagent/worker.py` | **Must `await load_repo_context(...)`** — see Deviation in Task 5. Also: plan calls `db.repo_attempts_today(...)` and `db.cost_today()` etc.; verify these exist in `src/ossagent/db.py` (they do — see Phase 1 file). |
| 14 | Telegram webhook + `push_and_open_pr` | `src/ossagent/webhook.py`, `src/ossagent/pr_creator.py`. Also update `make_send_tg_node` from Task 11 to write the sidecar. | The `gh` CLI must be installed in the Modal image (Task 16 handles this). |
| 15 | Crons (PR sync + stale cleanup) | `src/ossagent/crons.py` | Uses `PyGithub` (already in deps). |
| 16 | Modal app wiring | `src/ossagent/app.py` (rewrite) | **Deviation needed:** the plan uses deprecated `modal.Mount.from_local_dir(...)` and `mounts=[...]`. The existing `app.py` already uses the newer `image = image.add_local_dir("config", remote_path="/app/config").add_local_python_source("ossagent")` pattern — preserve that pattern. Also `@modal.web_endpoint` may be `@modal.fastapi_endpoint` in Modal 0.66+; check `import modal; help(modal.web_endpoint)` if it errors. |
| 17 | Deploy + register Telegram webhook | manual | **Stop here and let the user run it.** Requires `modal deploy`, then setting the Telegram webhook URL via curl, then a real issue URL for smoke test. Do not run these — they touch live Modal deployment and Telegram. |
| 18 | Phase 2 README | `README.md` (rewrite) | The plan's README in §18 is good as-is. |

---

## Project structure now

```
mend-pilot/
├── pyproject.toml                  # ← Task 1 added langgraph, tree-sitter, PyGithub
├── uv.lock                         # ← regenerated in Task 1
├── README.md                       # ← Phase 1, will be rewritten in Task 18
├── config/
│   ├── models.yaml
│   └── watched_repos.yaml
├── docs/superpowers/
│   ├── PHASE-2-HANDOFF.md          # ← THIS FILE
│   ├── plans/
│   │   ├── 2026-05-16-phase-1-skeleton.md
│   │   ├── 2026-05-16-phase-2-working-agent.md   # the plan being executed
│   │   └── 2026-05-17-phase-3-eval-and-polish.md
│   └── specs/
│       └── 2026-05-16-mend-pilot-design.md
└── src/ossagent/
    ├── __init__.py
    ├── agent/                      # ← entire subpackage is new in Phase 2
    │   ├── __init__.py             # Task 2
    │   ├── state.py                # Task 2 (TypedDict + dataclasses)
    │   ├── tools.py                # Task 3 (git/diff/ripgrep/windowing)
    │   ├── ast_locator.py          # Task 4 (tree-sitter)
    │   ├── context.py              # Task 5 (RepoContext + cache). load_repo_context is ASYNC.
    │   ├── context_extractor.py    # Task 6 (LLM extractor)
    │   └── nodes.py                # Tasks 7,8,9,10 → 11 (append). NodeFn type alias.
    ├── app.py                      # ← will be rewritten in Task 16
    ├── config.py
    ├── db.py
    ├── github_client.py
    ├── models.py
    ├── scheduler.py
    ├── telegram.py                 # ← Task 11 adds send_draft_for_approval method
    ├── telemetry.py
    └── triager.py
```

To-be-added in remaining tasks:
- `src/ossagent/agent/graph.py` (Task 12)
- `src/ossagent/worker.py` (Task 13)
- `src/ossagent/webhook.py` (Task 14)
- `src/ossagent/pr_creator.py` (Task 14)
- `src/ossagent/crons.py` (Task 15)

---

## Deviations from the plan — keep them

1. **`load_repo_context` is async** (Task 5). The plan declares it `def`, but the extractor returned by `make_extractor` is `async def`, so calling it synchronously returns an un-awaited coroutine that the bare `except Exception` swallows. In `worker.py` (Task 13), `repo_context = await load_repo_context(...)` instead of the plan's plain call.

2. **`TYPE_CHECKING` guard for `RepoContext` in `state.py`** (Task 2). The plan's `repo_context: "RepoContext"` string forward-ref is stripped by ruff. Using `if TYPE_CHECKING: from ossagent.agent.context import RepoContext` lets both ruff and mypy-strict pass.

3. **`NodeFn` type alias in `nodes.py`** (Task 7+). Every `make_*_node` factory has an explicit `-> NodeFn` return type. mypy-strict rejects un-annotated factories.

4. **`_as_text(content)` helper** (Task 7+). Because `LLM.ainvoke(...).content` is typed `str | list[ContentBlock]`, mypy-strict refuses to pass it directly to `json.loads(...)`. The helper coerces to `str`.

5. **`uv.lock` is committed** (Task 1). `uv pip install` does not regenerate the lockfile; `uv lock` does. The plan only mentions `uv pip install -e ".[dev]"` — also run `uv lock` and stage the diff.

6. **Modal image pattern** (Task 16, when reached). Preserve the existing `image.add_local_dir(...).add_local_python_source("ossagent")` pattern in `src/ossagent/app.py`. Do **not** introduce `modal.Mount(...)` from the plan — Modal deprecated it.

---

## Commit-message style

Follow the existing style: conventional-commits, lowercase subject, no trailing period. Examples already in this branch:
```
feat(agent): tree-sitter AST locator for Python symbols
feat(agent): enforce_style + run_tests nodes
chore: Phase 2 deps (langgraph, tree-sitter, PyGithub)
```

Do **not** add any of:
- `Co-Authored-By: Claude <...>`
- `🤖 Generated with Claude Code`
- `Generated-By:` trailers
- Any string mentioning Claude, AI, assistant, Anthropic, or LLM in commit metadata.

---

## Pre-commit gauntlet to expect

`.pre-commit-config.yaml` runs ruff, ruff-format, mypy in strict mode. mypy-strict tends to flag:
- Untyped function returns on factory closures → add `NodeFn` return type
- `msg.content` passed to `json.loads` → wrap in `_as_text(...)`
- `state.get("foo") or [None]` followed by indexing → use a separate `targets = state.get(...) or []` and a typed `Optional[X]` variable
- `subprocess.run(...)` without `capture_output` flags → fine as long as not asserting on stdout

After a commit fails due to ruff auto-fix, just re-stage and re-commit — the fix has already been applied.

---

## Quick sanity check for next session

```bash
cd /Users/kbtg/codebase/personal/mend-pilot
.venv/bin/python -c "
from ossagent.agent.state import AgentState, PRMetadata, Classification
from ossagent.agent.tools import shallow_clone_or_pull, apply_unified_diff
from ossagent.agent.ast_locator import find_python_symbol
from ossagent.agent.context import RepoContext, load_repo_context
from ossagent.agent.context_extractor import make_extractor
from ossagent.agent.nodes import (
    make_plan_node, make_locate_node, make_reproduce_node,
    make_implement_node, make_add_test_node,
    make_enforce_style_node, make_run_tests_node,
)
import inspect, asyncio
assert inspect.iscoroutinefunction(load_repo_context), 'load_repo_context must be async'
print('phase 2 state ok through task 10')
"
```

Expected output: `phase 2 state ok through task 10`.
