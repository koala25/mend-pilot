"""Thin async GitHub REST client."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class Issue:
    number: int
    title: str
    body: str
    labels: list[str]
    author: str
    html_url: str
    state: str
    assignee: str | None
    comments: int
    created_at: str


class GitHubClient:
    BASE = "https://api.github.com"

    def __init__(self, token: str | None = None) -> None:
        if token is None:
            try:
                token = os.environ["GITHUB_TOKEN"]
            except KeyError as e:
                raise RuntimeError("GITHUB_TOKEN is not set") from e
        self.token = token

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def fetch_new_issues(self, owner: str, name: str, *, since: str) -> list[Issue]:
        url = f"{self.BASE}/repos/{owner}/{name}/issues"
        params = {
            "state": "open",
            "sort": "created",
            "direction": "desc",
            "per_page": "30",
            "since": since,
        }
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url, headers=self._headers(), params=params)
            r.raise_for_status()
            data = r.json()
        return [self._to_issue(d) for d in data if "pull_request" not in d]

    async def fetch_issue(self, owner: str, name: str, number: int) -> Issue:
        url = f"{self.BASE}/repos/{owner}/{name}/issues/{number}"
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url, headers=self._headers())
            r.raise_for_status()
            return self._to_issue(r.json())

    @staticmethod
    def _to_issue(d: dict[str, Any]) -> Issue:
        return Issue(
            number=d["number"],
            title=d["title"],
            body=d.get("body") or "",
            labels=[lbl["name"] for lbl in d.get("labels", [])],
            author=d["user"]["login"],
            html_url=d["html_url"],
            state=d["state"],
            assignee=(d.get("assignee") or {}).get("login"),
            comments=d.get("comments", 0),
            created_at=d["created_at"],
        )
