"""Outbound Telegram notifications. Webhook handling is added in Phase 2."""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class TriageNotification:
    issue_url: str
    issue_title: str
    classification: str
    confidence: float
    reason: str


class TelegramBot:
    def __init__(self, token: str | None = None, user_id: str | None = None) -> None:
        if token is None:
            try:
                token = os.environ["TELEGRAM_BOT_TOKEN"]
            except KeyError as e:
                raise RuntimeError("TELEGRAM_BOT_TOKEN is not set") from e
        if user_id is None:
            try:
                user_id = os.environ["TELEGRAM_USER_ID"]
            except KeyError as e:
                raise RuntimeError("TELEGRAM_USER_ID is not set") from e
        self.token = token
        self.user_id = user_id

    async def send_triage_notification(self, n: TriageNotification) -> int:
        text = (
            f"🔍 New fit issue\n\n"
            f"<b>{_escape(n.issue_title)}</b>\n"
            f"{n.issue_url}\n\n"
            f"Classification: <b>{n.classification}</b>\n"
            f"Confidence: <b>{n.confidence:.2f}</b>\n"
            f"Reason: {_escape(n.reason)}"
        )
        return await self._send(text)

    async def _send(self, text: str) -> int:
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.user_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(url, json=payload)
            if r.status_code != 200:
                raise RuntimeError(f"Telegram API failed: {r.status_code} {r.text}")
            return int(r.json()["result"]["message_id"])


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
