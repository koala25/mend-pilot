# ossagent

A multi-agent open-source contribution bot. **Phase 1: skeleton — polling and triage only.**

## Status

Phase 1 deployed. Every 30 minutes the bot polls [`langchain-ai/langchain`](https://github.com/langchain-ai/langchain), classifies new issues with a cheap LLM (Moonshot's `moonshot-v1-8k`), and sends a Telegram message for any issue classified as a good candidate for automated fixing.

Subsequent phases will add the LangGraph agent that drafts the fixes themselves. See
[`docs/superpowers/specs/2026-05-16-oss-pr-bot-design.md`](docs/superpowers/specs/2026-05-16-oss-pr-bot-design.md)
for the full design, and [`docs/superpowers/plans/`](docs/superpowers/plans/) for the per-phase
implementation plans.

## Architecture (Phase 1)

```
GitHub Issues
   │  (cron, every 30m)
   ▼
poll_repos  ──►  Triager (Moonshot LLM)  ──►  Telegram (when fit + conf >= 0.7)
   │                                              │
   ▼                                              ▼
SQLite on Modal volume                       User's phone
(repo_state, cost_ledger, attempts)
```

Components — one responsibility each:

| File | Responsibility |
|---|---|
| `src/ossagent/config.py` | Load `models.yaml` + `watched_repos.yaml` |
| `src/ossagent/models.py` | Provider-agnostic LLM factory (Moonshot, OpenAI, Anthropic) |
| `src/ossagent/db.py` | SQLite schema + helpers (attempts, cost ledger, repo state); UTC-safe datetime adapters |
| `src/ossagent/telemetry.py` | `CostTracker` LangChain callback for $/token logging |
| `src/ossagent/github_client.py` | Async REST client (`fetch_new_issues`, `fetch_issue`) |
| `src/ossagent/telegram.py` | Outbound Telegram notifications (HTML-escaped) |
| `src/ossagent/triager.py` | Stage-1 LLM classifier → `TriageVerdict` |
| `src/ossagent/scheduler.py` | Per-tick orchestration: fetch → triage → notify |
| `src/ossagent/app.py` | Modal app entry point |

## Stack

- **Python 3.12**, [`uv`](https://docs.astral.sh/uv/) for env management
- **[Modal](https://modal.com)** for serverless cron + container deployment
- **[LangChain](https://python.langchain.com/)** core abstractions, model-agnostic via YAML config
- **[Moonshot](https://platform.moonshot.ai/)** for cheap fast triage; swappable to Anthropic / OpenAI by editing one YAML
- **SQLite** on a Modal volume for state
- **Telegram Bot API** for notifications
- **ruff + mypy strict** gating commits via pre-commit

## Cost (Phase 1)

Triager runs `moonshot-v1-8k`: $0.12/1M tokens both directions. At ~500 tokens per call and ~30 new issues/day in `langchain-ai/langchain`, the steady-state cost is **~$0.001/day**. The hard cap in the scheduler design is $50/month with a $3 per-attempt kill switch (active in Phase 2 once the heavy worker is wired in).

## Running locally

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
.venv/bin/pre-commit install
```

To run a one-shot poll outside Modal you'd need to mock the Modal volume — easier to just deploy.

## Deploying

```bash
# secrets — once
modal secret create ossagent-secrets --from-dotenv ~/.config/ossagent.env

# deploy or re-deploy
.venv/bin/modal deploy src/ossagent/app.py

# manual trigger (otherwise the cron fires every 30min)
.venv/bin/modal run src/ossagent/app.py::poll_repos
```

The expected `~/.config/ossagent.env`:

```
MOONSHOT_API_KEY=<from platform.moonshot.ai>
GITHUB_TOKEN=<fine-grained PAT with public_repo + issues:read>
TELEGRAM_BOT_TOKEN=<from @BotFather>
TELEGRAM_USER_ID=<your numeric id from @userinfobot>
```

**One-time Telegram setup:** the bot can only DM you after you've messaged it first. Open Telegram, find your bot, tap **Start**.

## Inspecting state

```bash
# Pull a copy of the SQLite database from the Modal volume
.venv/bin/modal volume get ossagent-data /attempts.db /tmp/attempts.db --force

# Watermark per repo
sqlite3 /tmp/attempts.db "SELECT * FROM repo_state;"

# Cost ledger (populated in Phase 2 when CostTracker is attached)
sqlite3 /tmp/attempts.db "SELECT role, sum(input_tokens), sum(output_tokens), sum(cost_usd) FROM cost_ledger GROUP BY role;"
```

## What's next (Phase 2+)

See [`docs/superpowers/plans/2026-05-16-phase-2-full-agent-system.md`](docs/superpowers/plans/2026-05-16-phase-2-full-agent-system.md):

1. LangGraph agent that proposes a fix for each fit issue
2. Telegram inline-keyboard approval flow → `gh pr create --draft`
3. Eval harness with single-shot baseline → headline solve-rate numbers
4. README updated with real merge-rate metrics from the eval set

## Repository layout

```
oss-pr-bot/
├── config/                    Provider + watched-repo configs (YAML)
├── src/ossagent/              Module-per-responsibility Python package
├── docs/superpowers/specs/    Design spec
├── docs/superpowers/plans/    Per-phase implementation plans (Phase 1 + Phase 2)
├── pyproject.toml             Deps, ruff/mypy config
├── .pre-commit-config.yaml    ruff + mypy hooks
└── uv.lock                    Pinned transitive dependency versions
```
