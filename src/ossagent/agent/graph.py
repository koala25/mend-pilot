"""Build and compile the LangGraph state machine with all 4 classifications."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ossagent.agent.nodes import (
    make_add_test_node,
    make_critic_node,
    make_enforce_style_node,
    make_implement_node,
    make_locate_node,
    make_plan_node,
    make_prepare_pr_node,
    make_reproduce_node,
    make_run_tests_node,
    make_send_tg_node,
)
from ossagent.agent.state import AgentState
from ossagent.telegram import TelegramBot

MAX_RETRIES = 2
MAX_STYLE_RETRIES = 2
MAX_ATTEMPT_BUDGET = 3.00


def build_graph(
    *,
    llms: dict[str, BaseChatModel],
    telegram_bot: TelegramBot,
    our_login: str,
    checkpoint_db: Path,
) -> CompiledStateGraph:
    g: StateGraph = StateGraph(AgentState)

    g.add_node("plan", make_plan_node(llms["planner"]))
    g.add_node("locate", make_locate_node(llms["locator"]))
    g.add_node("reproduce", make_reproduce_node(llms["implementer"]))
    g.add_node("implement", make_implement_node(llms["implementer"]))
    g.add_node("add_test", make_add_test_node(llms["tester"]))
    g.add_node("enforce_style", make_enforce_style_node())
    g.add_node("run_tests", make_run_tests_node())
    g.add_node("critic", make_critic_node(llms["critic"]))
    g.add_node("prepare_pr", make_prepare_pr_node(llms["pr_writer"], our_login=our_login))
    g.add_node("send_tg", make_send_tg_node(telegram_bot))
    g.add_node("log_skip", _noop)

    g.set_entry_point("plan")
    g.add_edge("plan", "locate")
    g.add_conditional_edges(
        "locate",
        _route_after_locate,
        {"reproduce": "reproduce", "implement": "implement", "log_skip": "log_skip"},
    )
    g.add_edge("reproduce", "implement")
    g.add_conditional_edges(
        "implement",
        _route_after_implement,
        {"add_test": "add_test", "enforce_style": "enforce_style", "log_skip": "log_skip"},
    )
    g.add_edge("add_test", "enforce_style")
    g.add_conditional_edges(
        "enforce_style",
        _route_after_style,
        {"run_tests": "run_tests", "implement": "implement", "log_skip": "log_skip"},
    )
    g.add_edge("run_tests", "critic")
    g.add_conditional_edges(
        "critic",
        _decide_next,
        {"prepare_pr": "prepare_pr", "plan": "plan", "log_skip": "log_skip"},
    )
    g.add_edge("prepare_pr", "send_tg")
    g.add_edge("send_tg", END)
    g.add_edge("log_skip", END)

    checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(checkpoint_db), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return g.compile(checkpointer=checkpointer)


def _route_after_locate(state: AgentState) -> str:
    if not state.get("target_files"):
        return "log_skip"
    if state.get("classification") == "BUG_FIX":
        return "reproduce"
    return "implement"


def _route_after_implement(state: AgentState) -> str:
    if not state.get("patch"):
        return "log_skip"
    if state.get("classification") == "TEST_GAP":
        return "add_test"
    return "enforce_style"


def _route_after_style(state: AgentState) -> str:
    if not state.get("style_violations"):
        return "run_tests"
    if state.get("style_retry_count", 0) < MAX_STYLE_RETRIES:
        return "implement"
    return "log_skip"


def _decide_next(state: AgentState) -> str:
    if state.get("cost_so_far", 0.0) > MAX_ATTEMPT_BUDGET:
        return "log_skip"
    verdict = state.get("critic_verdict")
    if verdict == "ABORT":
        return "log_skip"
    if verdict == "PASS":
        return "prepare_pr"
    if verdict == "RETRY" and state.get("retry_count", 0) < MAX_RETRIES:
        return "plan"
    return "log_skip"


async def _noop(state: AgentState) -> dict[str, Any]:
    return {}
