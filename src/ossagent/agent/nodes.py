"""LangGraph node functions. Each takes AgentState and returns partial updates."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ossagent.agent.ast_locator import find_python_symbol
from ossagent.agent.context import RepoContext
from ossagent.agent.state import AgentState, PlanStep, TargetFile
from ossagent.agent.tools import read_file_window, ripgrep

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
