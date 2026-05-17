"""LangGraph node functions. Each takes AgentState and returns partial updates."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ossagent.agent.ast_locator import find_python_symbol
from ossagent.agent.context import RepoContext
from ossagent.agent.state import (
    AgentState,
    PlanStep,
    PRMetadata,
    StyleViolation,
    TargetFile,
    TestRunResult,
)
from ossagent.agent.tools import apply_unified_diff, read_file_window, ripgrep
from ossagent.telegram import TelegramBot

NodeFn = Callable[[AgentState], Awaitable[dict[str, Any]]]


def _as_text(content: Any) -> str:
    return content if isinstance(content, str) else str(content)


# ── plan node ───────────────────────────────────────────────────────────

PLAN_SYSTEM = """You are the Planner. Given a GitHub issue and repo conventions,
output a SHORT numbered plan (<=4 steps) for a single-file fix. If multi-file
is needed, plan with one step: "ABORT: multi-file refactor".

Reply STRICT JSON: {"steps": [{"n": 1, "d": "<step>"}, ...]}
"""


def make_plan_node(llm: BaseChatModel) -> NodeFn:
    async def plan_node(state: AgentState) -> dict[str, Any]:
        issue = state["issue"]
        ctx: RepoContext = state["repo_context"]
        style_summary = "\n".join(f"- {n}" for n in ctx.style_notes[:6]) or "(none extracted)"
        user = (
            f"# Classification: {state['classification']}\n\n"
            f"# Issue\n{issue.title}\n\n{issue.body[:3000]}\n\n"
            f"# Repo style notes\n{style_summary}\n\n"
            f"# Contributing excerpt\n{(ctx.contributing_md or '')[:2000]}\n"
        )
        msg = await llm.ainvoke(
            [
                SystemMessage(content=PLAN_SYSTEM),
                HumanMessage(content=user),
            ]
        )
        try:
            data = json.loads(_as_text(msg.content))
            steps = [PlanStep(number=int(s["n"]), description=str(s["d"])) for s in data["steps"]]
        except (json.JSONDecodeError, KeyError, ValueError):
            steps = []
        return {"plan": steps}

    return plan_node


# ── locate node ─────────────────────────────────────────────────────────

LOCATE_SYSTEM = """You are the Locator. Given the plan, repo, and search hits,
identify the SINGLE file + line range to modify.

You will receive:
  - Ripgrep matches for keywords from the issue
  - AST symbol hits for any identifier-like tokens
  - File excerpts

Reply STRICT JSON:
{"target": {"path": "<rel path>", "line_start": <int>,
            "line_end": <int>, "why": "<one sentence>"}}

If no single-file location works:
{"target": null, "reason": "<reason>"}
"""


def make_locate_node(llm: BaseChatModel) -> NodeFn:
    async def locate_node(state: AgentState) -> dict[str, Any]:
        body = state["issue"].body[:2000]
        keywords = _extract_keywords(body)
        rg_lines: list[str] = []
        ast_hits: list[str] = []
        for kw in keywords[:5]:
            rg_lines.extend(ripgrep(state["repo_path"], kw, max_results=6))
            for sh in find_python_symbol(state["repo_path"], kw, max_results=3):
                ast_hits.append(
                    f"{sh.path}:{sh.line_start}-{sh.line_end} {sh.kind} {sh.symbol}",
                )
        plan_text = "\n".join(f"{s.number}. {s.description}" for s in state.get("plan", []))
        user = (
            f"# Plan\n{plan_text}\n\n"
            f"# Issue title\n{state['issue'].title}\n\n"
            f"# Ripgrep hits\n{chr(10).join(rg_lines[:25]) or '(no matches)'}\n\n"
            f"# AST hits\n{chr(10).join(ast_hits[:15]) or '(no matches)'}\n"
        )
        msg = await llm.ainvoke(
            [
                SystemMessage(content=LOCATE_SYSTEM),
                HumanMessage(content=user),
            ]
        )
        try:
            data = json.loads(_as_text(msg.content))
            t = data.get("target")
            if t is None:
                return {"target_files": [], "skip_reason": data.get("reason", "no_location")}
            return {
                "target_files": [
                    TargetFile(
                        path=t["path"],
                        line_range=(int(t["line_start"]), int(t["line_end"])),
                        why=t["why"],
                    )
                ]
            }
        except (json.JSONDecodeError, KeyError, ValueError):
            return {"target_files": [], "skip_reason": "locate_parse_failed"}

    return locate_node


def _extract_keywords(text: str) -> list[str]:
    quoted = re.findall(r"`([^`]+)`", text)
    idents = re.findall(r"\b([A-Z][A-Za-z0-9_]{3,}|[a-z_][a-z0-9_]{4,})\b", text)
    seen: set[str] = set()
    out: list[str] = []
    for k in quoted + idents:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


# ── reproduce node (BUG_FIX only) ───────────────────────────────────────

REPRODUCE_SYSTEM = """You are the Reproducer. Given a bug report, write a
pytest test that DEMONSTRATES the bug — it must FAIL on the current code.

Output STRICT JSON:
{"test_path": "tests/<test_filename>.py",
 "test_code": "<full file contents>"}
"""


def make_reproduce_node(llm: BaseChatModel) -> NodeFn:
    async def reproduce_node(state: AgentState) -> dict[str, Any]:
        issue = state["issue"]
        targets = state.get("target_files") or []
        target: TargetFile | None = targets[0] if targets else None
        target_summary = (
            "(none)"
            if target is None
            else f"{target.path} lines {target.line_range[0]}-{target.line_range[1]}: {target.why}"
        )
        user = (
            f"# Bug report\n{issue.title}\n\n{issue.body[:3000]}\n\n"
            f"# Target file\n{target_summary}\n"
        )
        msg = await llm.ainvoke(
            [
                SystemMessage(content=REPRODUCE_SYSTEM),
                HumanMessage(content=user),
            ]
        )
        try:
            data = json.loads(_as_text(msg.content))
            test_path = str(data["test_path"])
            test_code = str(data["test_code"])
        except (json.JSONDecodeError, KeyError, ValueError):
            return {"skip_reason": "reproduce_parse_failed"}

        repo_path = state["repo_path"]
        target_path = repo_path / test_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(test_code)

        proc = subprocess.run(
            ["pytest", "-x", "--timeout=60", "-q", test_path],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if proc.returncode == 0:
            # Test passed — bug NOT demonstrated. ABORT.
            target_path.unlink(missing_ok=True)
            return {"skip_reason": "could_not_reproduce_bug"}
        return {
            "failing_test_path": test_path,
            "failing_test_output": (proc.stdout or "")[-2000:],
        }

    return reproduce_node


# ── implement node ──────────────────────────────────────────────────────

IMPLEMENT_SYSTEM = """You are the Implementer. Output a UNIFIED DIFF that fixes
the issue. Rules:
- Output ONLY the diff, no prose, no code fences.
- Single file. Surgical changes.
- Preserve indentation.
- Must apply cleanly with `git apply`.
"""


def make_implement_node(llm: BaseChatModel) -> NodeFn:
    async def implement_node(state: AgentState) -> dict[str, Any]:
        if not state.get("target_files"):
            return {"skip_reason": "no_target_for_implement"}
        target = state["target_files"][0]
        repo_path = state["repo_path"]
        window = read_file_window(
            repo_path,
            target.path,
            line=target.line_range[0],
            before=20,
            after=80,
        )
        ctx: RepoContext = state["repo_context"]
        plan_text = "\n".join(f"{s.number}. {s.description}" for s in state.get("plan", []))
        style_notes = "\n".join(f"- {n}" for n in ctx.style_notes[:6]) or "(none)"
        failing_test_section = ""
        if state.get("failing_test_output"):
            failing_test_section = (
                "\n# Failing test (must pass after your fix)\n"
                f"{(state['failing_test_output'] or '')[-1500:]}\n"
            )
        user = (
            f"# Plan\n{plan_text}\n\n"
            f"# Target\n{target.path} L{target.line_range[0]}-{target.line_range[1]} :: {target.why}\n\n"
            f"# File window\n```\n{window}\n```\n\n"
            f"# Style notes\n{style_notes}\n"
            f"{failing_test_section}"
        )
        msg = await llm.ainvoke(
            [
                SystemMessage(content=IMPLEMENT_SYSTEM),
                HumanMessage(content=user),
            ]
        )
        text = _as_text(msg.content).strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
            text = text.rsplit("```", 1)[0]
        return {"patch": text.strip()}

    return implement_node


# ── add_test node (TEST_GAP only) ───────────────────────────────────────

ADD_TEST_SYSTEM = """You are the Test Writer. Write a pytest test (or tests)
covering the public symbol(s) the issue identifies as untested.

Output STRICT JSON:
{"test_path": "tests/<file>.py",
 "test_code": "<full file contents>",
 "append_only": true | false}    // true = append to existing file
"""


def make_add_test_node(llm: BaseChatModel) -> NodeFn:
    async def add_test_node(state: AgentState) -> dict[str, Any]:
        ctx: RepoContext = state["repo_context"]
        target = state["target_files"][0]
        user = (
            f"# Symbol(s) to cover\n{target.path}:{target.line_range[0]}-{target.line_range[1]}\n"
            f"why: {target.why}\n\n"
            f"# Sample test file\n{ctx.sample_test_excerpt}\n\n"
            f"# Test patterns\n{chr(10).join('- ' + p for p in ctx.test_patterns[:4]) or '(none)'}\n"
        )
        msg = await llm.ainvoke(
            [
                SystemMessage(content=ADD_TEST_SYSTEM),
                HumanMessage(content=user),
            ]
        )
        try:
            data = json.loads(_as_text(msg.content))
        except (json.JSONDecodeError, KeyError, ValueError):
            return {"skip_reason": "add_test_parse_failed"}

        repo_path = state["repo_path"]
        tp = repo_path / data["test_path"]
        tp.parent.mkdir(parents=True, exist_ok=True)
        if data.get("append_only") and tp.exists():
            with tp.open("a") as f:
                f.write("\n\n" + data["test_code"])
        else:
            tp.write_text(data["test_code"])
        return {}

    return add_test_node


# ── enforce_style node (deterministic) ──────────────────────────────────


def make_enforce_style_node() -> NodeFn:
    async def enforce_style_node(state: AgentState) -> dict[str, Any]:
        repo_path = state["repo_path"]
        patch = state.get("patch", "")
        if not patch:
            return {"style_violations": [], "skip_reason": "no_patch"}
        try:
            apply_unified_diff(repo_path, patch)
        except RuntimeError as e:
            return {
                "style_violations": [
                    StyleViolation(tool="git-apply", file="(diff)", message=str(e))
                ],
                "style_retry_count": state.get("style_retry_count", 0) + 1,
            }
        diff_files = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        violations: list[StyleViolation] = []
        if diff_files:
            subprocess.run(
                ["ruff", "check", "--fix", "--exit-zero", *diff_files],
                cwd=repo_path,
                timeout=60,
            )
            subprocess.run(["ruff", "format", *diff_files], cwd=repo_path, timeout=60)
            remaining = subprocess.run(
                ["ruff", "check", *diff_files],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if remaining.returncode != 0:
                for line in (remaining.stdout or "").splitlines():
                    if ":" in line:
                        path, _, message = line.partition(":")
                        violations.append(
                            StyleViolation(tool="ruff", file=path.strip(), message=message.strip())
                        )
        return {
            "style_violations": violations,
            "style_retry_count": state.get("style_retry_count", 0) + (1 if violations else 0),
        }

    return enforce_style_node


# ── run_tests node ──────────────────────────────────────────────────────


def make_run_tests_node() -> NodeFn:
    async def run_tests_node(state: AgentState) -> dict[str, Any]:
        repo_path = state["repo_path"]
        diff_files = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        targets: list[str] = []
        for f in diff_files:
            module = f.replace("/", ".").removesuffix(".py")
            targets.extend(_find_related_tests(repo_path, module))
        # Always include the failing-test path if reproducer wrote one.
        failing = state.get("failing_test_path")
        if failing:
            targets.append(failing)
        if not targets:
            targets = ["tests/"]
        proc = subprocess.run(
            ["pytest", "-x", "--timeout=60", "-q", *targets],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=300,
        )
        passed = proc.returncode == 0
        failed = [line.strip() for line in (proc.stdout or "").splitlines() if "FAILED" in line]
        return {
            "test_run": TestRunResult(
                passed=passed,
                failed_tests=failed[:10],
                stdout_tail=(proc.stdout or "")[-2000:],
                stderr_tail=(proc.stderr or "")[-1000:],
            )
        }

    return run_tests_node


def _find_related_tests(repo_path: Any, module_dotted: str) -> list[str]:
    base = module_dotted.split(".")[-1]
    if not base:
        return []
    return [str(p.relative_to(repo_path)) for p in repo_path.glob(f"tests/**/test_{base}.py")]


# ── critic node ─────────────────────────────────────────────────────────

CRITIC_SYSTEM = """You are the Critic. Review the proposed change adversarially.

Decide ONE of: PASS / RETRY / ABORT.

Reply STRICT JSON:
{"verdict": "PASS|RETRY|ABORT",
 "confidence": 0.0-1.0,
 "reasoning": "<2-3 sentences>"}
"""


def make_critic_node(llm: BaseChatModel) -> NodeFn:
    async def critic_node(state: AgentState) -> dict[str, Any]:
        ctx: RepoContext = state["repo_context"]
        tr: TestRunResult = state["test_run"]
        violations = state.get("style_violations", [])
        v_summary = (
            "(none)"
            if not violations
            else "\n".join(f"- [{v.tool}] {v.file}: {v.message}" for v in violations[:10])
        )
        user = (
            f"# Issue\n{state['issue'].title}\n\n{state['issue'].body[:2000]}\n\n"
            f"# Diff\n```\n{state.get('patch', '')[:8000]}\n```\n\n"
            f"# Tests\npassed={tr.passed} failed={tr.failed_tests}\n"
            f"stdout_tail:\n{tr.stdout_tail[-1000:]}\n\n"
            f"# Style violations\n{v_summary}\n\n"
            f"# Conventions\n{(ctx.contributing_md or '')[:2000]}\n"
        )
        msg = await llm.ainvoke(
            [
                SystemMessage(content=CRITIC_SYSTEM),
                HumanMessage(content=user),
            ]
        )
        try:
            data = json.loads(_as_text(msg.content))
            verdict = str(data["verdict"])
            confidence = float(data["confidence"])
            reasoning = str(data["reasoning"])
        except (json.JSONDecodeError, KeyError, ValueError):
            verdict, confidence, reasoning = "ABORT", 0.0, "critic_parse_failed"
        # Hard overrides
        if not tr.passed:
            verdict = "RETRY"
        if violations and verdict == "PASS":
            verdict = "RETRY"
        if confidence < 0.6 and verdict == "PASS":
            verdict = "RETRY"
        return {
            "critic_verdict": verdict,
            "critic_reasoning": reasoning,
            "confidence": confidence,
        }

    return critic_node


# ── prepare_pr node ─────────────────────────────────────────────────────
#
# Two LLM passes. First writes the PR; second rewrites the body to strip
# AI-tell stylistics. NEVER mention this project, "agent", "automation", or
# similar in the output — author the PR as the human reviewer.


PR_WRITER_SYSTEM = """Write the title and body for a GitHub pull request.

You will be given: the issue this PR addresses, the diff, test results, and
(if present) the repo's PR template.

VOICE — read carefully. The PR should read as if a senior developer wrote it
after thirty minutes of investigation. Be specific and technical. Skip filler.

NEVER include any of these words anywhere in title or body:
  automated, agent, bot, AI, LLM, drafted, generated, Claude, Kimi, Moonshot,
  ossagent, mend-pilot, assistant. Do not refer to "the tool" or "this script"
  as the author of the change. The reviewer is the author.

NEVER use these stylistic crutches:
  Em dashes. Use commas, periods, or parentheses.
  Words: leverage, robust, comprehensive, delve, underscore, showcase, ensure,
    foster, seamless, vital, crucial, pivotal, harness, streamline.
  -ing tails adding fake depth ("highlighting that", "ensuring X").
  "Not only X but Y" or "It's not just about X, it's Y".
  Triple lists where two would do.
  Emojis.
  Bolded inline headers like **Summary:** or **Test plan:** in body prose.
  Sycophantic closers ("I hope this helps", "Happy to iterate").
  Curly quotes.
  Title Case Section Headings (use sentence case).

CONTENT the body MUST include:
  - A link to the issue. Use `Closes #N` if same repo, full URL otherwise.
  - One sentence describing what the diff does.
  - Two to four sentences on WHY this fix and not an alternative: the root
    cause, what made it surface, what else was considered and rejected.
  - A short concrete verification section: which existing or new tests cover
    this, and the exact pytest invocation that was run.
  - If the repo's PR template is provided, fill ITS sections; do not invent
    headings that aren't in the template.

TITLE: conventional-commits style, lowercase subject, no trailing period.
Examples:
  fix(runnables): replace deprecated Runnable.run() with .invoke()
  docs: correct typo in tool-calling guide

BODY length: 80-250 words. Less is fine for trivial changes.

VOICE cues:
  Short, declarative sentences.
  First person is fine: "I noticed", "I tried X first", "I think".
  Specifics over abstractions. Function names, file paths, error messages,
    test names beat generic claims about "the change".

Reply STRICT JSON: {"title": "<title>", "body": "<markdown body>"}
"""


HUMANIZER_SYSTEM = """You are an editor stripping signs of AI authorship from
a draft pull-request description. Apply these rewrites:

  Replace any em dashes with commas, periods, or parentheses.
  Remove or rephrase to drop these words: leverage, robust, comprehensive,
    delve, underscore, showcase, ensure, foster, seamless, vital, crucial,
    pivotal, harness, streamline, automated, AI, agent, bot, drafted,
    generated, Claude, Kimi, Moonshot, ossagent, mend-pilot, assistant.
  Remove -ing tail clauses adding fake depth ("highlighting that", "ensuring
    X", "underscoring its importance").
  Remove "not only/but also" and "it's not just/it's" constructions.
  Collapse triple lists where two items already convey the point.
  Remove emoji.
  Convert bolded inline headers (**Foo:**) to plain prose.
  Remove sycophantic closers.
  Sentence-case any headings (not Title Case).
  Replace curly quotes with straight ones.

KEEP all technical content unchanged: file paths, function names, line
numbers, test names, error messages, the issue link, the description of what
the diff does. Do not invent or remove technical claims.

Reply with just the rewritten body. No JSON, no commentary, no code fences.
"""


def make_prepare_pr_node(llm: BaseChatModel, *, our_login: str) -> NodeFn:
    async def prepare_pr_node(state: AgentState) -> dict[str, Any]:
        issue = state["issue"]
        ctx: RepoContext = state["repo_context"]
        tr: TestRunResult = state["test_run"]

        # Pass 1 — write the PR.
        writer_input = (
            f"# Issue URL\n{issue.html_url}\n\n"
            f"# Issue title\n{issue.title}\n\n"
            f"# Issue body excerpt\n{issue.body[:1500]}\n\n"
            f"# Diff\n```\n{state.get('patch', '')[:6000]}\n```\n\n"
            f"# Tests\npassed={tr.passed} failed={tr.failed_tests}\n"
            f"stdout_tail:\n{tr.stdout_tail[-800:]}\n\n"
            f"# PR template (verbatim from repo, fill as-is if present)\n"
            f"{ctx.pr_template or '(none)'}\n\n"
            f"# PR norms\n"
            f"{chr(10).join('- ' + n for n in ctx.pr_norms[:4]) or '(none)'}\n"
        )
        write_msg = await llm.ainvoke(
            [
                SystemMessage(content=PR_WRITER_SYSTEM),
                HumanMessage(content=writer_input),
            ]
        )
        try:
            data = json.loads(_as_text(write_msg.content))
            title = str(data["title"])
            draft_body = str(data["body"])
        except (json.JSONDecodeError, KeyError, ValueError):
            # Generic minimal fallback. Never mentions our tooling.
            title = _fallback_title(issue.title)
            draft_body = f"Closes {issue.html_url}\n\nReviewed locally before submitting."

        # Pass 2 — humanizer pass on the body. The title is already
        # constrained to conventional-commits style by the writer prompt.
        humanize_msg = await llm.ainvoke(
            [
                SystemMessage(content=HUMANIZER_SYSTEM),
                HumanMessage(content=draft_body),
            ]
        )
        body = (_as_text(humanize_msg.content) or draft_body).strip()

        branch = f"fix/issue-{issue.number}-{state['attempt_id'][:8]}"
        base_branch = "master" if ctx.repo_owner in {"langchain-ai", "tiangolo"} else "main"
        return {
            "pr_metadata": PRMetadata(
                title=title,
                body=body,
                branch_name=branch,
                head_owner=our_login,
                base_branch=base_branch,
            )
        }

    return prepare_pr_node


def _fallback_title(issue_title: str) -> str:
    """Conventional-commits-style title derived from the issue title."""
    cleaned = issue_title.strip().rstrip(".").lower()[:60]
    return f"fix: {cleaned}"


# ── send_tg node ────────────────────────────────────────────────────────


def make_send_tg_node(telegram_bot: TelegramBot, *, data_dir: Path) -> NodeFn:
    async def send_tg_node(state: AgentState) -> dict[str, Any]:
        pr = state["pr_metadata"]
        attempt_id = state["attempt_id"]
        # Sidecar lets push_and_open_pr (separate Modal function) reconstruct
        # the prepared title, body, and base branch without needing the graph
        # checkpoint.
        sidecar_dir = data_dir / "drafts" / attempt_id
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        (sidecar_dir / "pr.json").write_text(
            json.dumps(
                {
                    "title": pr.title,
                    "body": pr.body,
                    "base_branch": pr.base_branch,
                    "branch_name": pr.branch_name,
                    "head_owner": pr.head_owner,
                },
                indent=2,
            )
        )
        await telegram_bot.send_draft_for_approval(
            attempt_id=attempt_id,
            issue_url=state["issue"].html_url,
            issue_title=state["issue"].title,
            classification=state["classification"],
            confidence=state.get("confidence", 0.0),
            critic_reasoning=state.get("critic_reasoning", ""),
            patch_excerpt=state.get("patch", "")[:1500],
            pr_title=pr.title,
        )
        return {}

    return send_tg_node
