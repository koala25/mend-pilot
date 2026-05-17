# mend-pilot

A multi-agent open-source contribution bot. **Phase 2: working agent — real draft PRs.**

## Status

Phase 2 deployed. The bot watches selected open-source repos, classifies new issues with a cheap LLM triager, runs a LangGraph agent to draft a fix for fit issues, asks for single-click approval via Telegram, and on approval pushes a branch and opens a GitHub **draft PR**.

The headline numbers (solve rate, $/PR, agent-vs-baseline lift) are produced in **Phase 3** ([plan](docs/superpowers/plans/2026-05-17-phase-3-eval-and-polish.md)).

## Architecture

```
GitHub Issues
   │  (cron, every 30m)
   ▼
poll_repos  ──►  Triager (Moonshot LLM)
                    │  (fit + conf >= 0.7, DEPRECATION/BUG_FIX/etc.)
                    ▼
              process_issue_fn (Modal worker)
                    │
                    ▼
              LangGraph: plan → locate → reproduce? → implement → add_test?
                          → enforce_style → run_tests → critic → prepare_pr → send_tg
                    │
                    ▼
              Telegram message with inline buttons
                    │  (you tap Approve)
                    ▼
              telegram_webhook → push_and_open_pr_fn
                    │
                    ▼
              GitHub draft PR on the upstream repo
```

Components — one responsibility each:

| File | Responsibility |
|---|---|
| `src/ossagent/agent/state.py` | `AgentState` TypedDict + supporting dataclasses |
| `src/ossagent/agent/context.py` | `RepoContext` loader with on-disk cache |
| `src/ossagent/agent/context_extractor.py` | LLM extraction of style notes, test patterns, PR norms |
| `src/ossagent/agent/tools.py` | git, diff, ripgrep, file-window helpers |
| `src/ossagent/agent/ast_locator.py` | tree-sitter symbol lookup for Python repos |
| `src/ossagent/agent/nodes.py` | All 11 LangGraph node functions |
| `src/ossagent/agent/graph.py` | `build_graph()` with conditional routing + checkpointing |
| `src/ossagent/worker.py` | `process_issue` with 8-step pre-process guards |
| `src/ossagent/webhook.py` | Telegram callback handler |
| `src/ossagent/pr_creator.py` | Commit + push + `gh pr create --draft` |
| `src/ossagent/crons.py` | Daily PR-status sync + stale-draft cleanup |
| `src/ossagent/app.py` | Modal app — 5 functions (scheduler, worker, webhook, PR opener, daily cron) |

## Operational cost

Based on Phase 1 actuals × Phase 2 multipliers. On `moonshot-v1-8k`:

- Triager: ~$0.05/mo (~9k cheap calls)
- Agent runs: ~$0.30-1.00/mo (~150 attempts × ~$0.003 each, 7-node pipeline)
- Modal compute: well under the $30 free credit
- **Realistic total: ~$0.50-1/month**

Hard caps in place: $3/attempt, $5/day, $50/month. Auto-pause at 95% of monthly cap.

## Running locally

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
.venv/bin/pre-commit install
```

To run end-to-end you need to deploy — local mocks for the Modal volume aren't part of v1.

## Deploying

```bash
modal secret create ossagent-secrets --from-dotenv ~/.config/ossagent.env --force
.venv/bin/modal deploy src/ossagent/app.py
```

Then register the Telegram webhook (see Task 17 of `docs/superpowers/plans/2026-05-16-phase-2-working-agent.md`).

## Inspecting state

```bash
.venv/bin/modal volume get ossagent-data /attempts.db /tmp/attempts.db --force
sqlite3 /tmp/attempts.db "SELECT attempt_id, status, repo_owner, classification, started_at FROM attempts ORDER BY started_at DESC LIMIT 20;"
sqlite3 /tmp/attempts.db "SELECT role, sum(input_tokens), sum(output_tokens), sum(cost_usd) FROM cost_ledger GROUP BY role;"
```

## What's next

- [Phase 3](docs/superpowers/plans/2026-05-17-phase-3-eval-and-polish.md) — eval harness, headline solve-rate vs single-shot baseline, architecture diagram, 60-second demo video, recruiter-grade README polish.
- Phase 2.5 — deferred unit-test pass (see the Phase 2 plan).
