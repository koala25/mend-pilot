"""LangGraph node functions. Each takes AgentState and returns partial updates."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ossagent.agent.ast_locator import find_python_symbol
from ossagent.agent.context import RepoContext
from ossagent.agent.state import AgentState, PlanStep, TargetFile
from ossagent.agent.tools import ripgrep

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
