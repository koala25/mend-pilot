"""LLM-driven extraction of style_notes, test_patterns, pr_norms, readme_summary."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

SYSTEM = """You read a repo's CONTRIBUTING.md and README, and extract specific
conventions for an automated contribution bot.

Reply STRICT JSON:
{
  "style_notes": ["...", ...],         // <= 6 short rules (e.g., "Google-style docstrings")
  "test_patterns": ["...", ...],       // <= 4 patterns (e.g., "pytest fixtures in conftest.py")
  "pr_norms": ["...", ...],            // <= 4 norms (e.g., "Title in conventional-commits format")
  "readme_summary": "..."              // <= 200 words
}
"""


Extractor = Callable[[str, str], Awaitable[tuple[list[str], list[str], list[str], str]]]


def make_extractor(llm: BaseChatModel) -> Extractor:
    async def extract(
        contributing: str, readme: str
    ) -> tuple[list[str], list[str], list[str], str]:
        user = f"# CONTRIBUTING.md\n{contributing[:6000]}\n\n" f"# README.md\n{readme[:4000]}\n"
        msg = await llm.ainvoke(
            [
                SystemMessage(content=SYSTEM),
                HumanMessage(content=user),
            ]
        )
        try:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            data = json.loads(content)
            return (
                list(data.get("style_notes", []))[:6],
                list(data.get("test_patterns", []))[:4],
                list(data.get("pr_norms", []))[:4],
                str(data.get("readme_summary", ""))[:1500],
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return [], [], [], readme[:1000]

    return extract
