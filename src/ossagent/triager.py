"""Stage 1 cheap LLM classifier. Decides if an issue is worth the heavy worker."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import StrEnum

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ossagent.github_client import Issue

log = logging.getLogger(__name__)


class Classification(StrEnum):
    TYPO = "TYPO"
    DEPRECATION = "DEPRECATION"
    TEST_GAP = "TEST_GAP"
    BUG_FIX = "BUG_FIX"
    UNCLASSIFIED = "UNCLASSIFIED"  # fallback when the LLM returns garbage


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
        msg = await self.llm.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ]
        )
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
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            log.warning(
                "triager parse failed: %s; raw=%r",
                e,
                text[:500],
            )
            return TriageVerdict(
                fit=False,
                confidence=0.0,
                classification=Classification.UNCLASSIFIED,
                reason="Triager returned malformed JSON; rejecting by default.",
            )
