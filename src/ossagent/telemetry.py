"""Cost-tracking LangChain callback that writes to the cost_ledger table."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from ossagent.db import CostLedgerEntry, Database

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPrice:
    input_per_1m: float
    output_per_1m: float


# Pin from provider docs at deploy time.
MODEL_PRICING: dict[str, ModelPrice] = {
    "kimi-latest": ModelPrice(input_per_1m=0.30, output_per_1m=1.50),
    "claude-haiku-4-5": ModelPrice(input_per_1m=0.80, output_per_1m=4.00),
    "claude-sonnet-4-6": ModelPrice(input_per_1m=3.00, output_per_1m=15.00),
    "claude-opus-4-7": ModelPrice(input_per_1m=15.00, output_per_1m=75.00),
}


class CostTracker(BaseCallbackHandler):  # type: ignore[misc]  # langchain BaseCallbackHandler typed as Any
    def __init__(self, db: Database, attempt_id: str, role: str, model: str) -> None:
        self.db = db
        self.attempt_id = attempt_id
        self.role = role
        self.model = model

    def on_llm_end(self, response: LLMResult, **kwargs: object) -> None:
        # Telemetry must never break the chain it observes — swallow & log any
        # storage failure so a transient sqlite issue can't derail an LLM call.
        try:
            usage = self._extract_usage(response)
            if usage is None:
                return
            input_tokens, output_tokens = usage
            price = MODEL_PRICING.get(self.model)
            if price is None:
                log.warning(
                    "unknown model %r; recording zero cost (update MODEL_PRICING)",
                    self.model,
                )
                cost_usd = 0.0
            else:
                cost_usd = (input_tokens / 1_000_000) * price.input_per_1m + (
                    output_tokens / 1_000_000
                ) * price.output_per_1m
            self.db.add_cost(
                CostLedgerEntry(
                    attempt_id=self.attempt_id,
                    role=self.role,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    at=datetime.now(UTC),
                )
            )
        except Exception:
            log.exception("cost telemetry failed for attempt=%s", self.attempt_id)

    @staticmethod
    def _extract_usage(response: LLMResult) -> tuple[int, int] | None:
        # Phase 1 assumption: one generation per call (Triager). Returns first hit.
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                um = getattr(msg, "usage_metadata", None)
                if um:
                    return int(um["input_tokens"]), int(um["output_tokens"])
        return None
