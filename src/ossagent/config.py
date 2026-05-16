"""Load and validate YAML configuration files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REQUIRED_ROLES = (
    "triager",
    "planner",
    "locator",
    "implementer",
    "tester",
    "critic",
    "pr_writer",
)


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    api_base: str | None
    model: str
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class WatchedRepo:
    owner: str
    name: str
    default_branch: str


def load_models_config(path: Path) -> dict[str, ModelSpec]:
    raw = yaml.safe_load(path.read_text())
    defaults = raw.get("defaults", {})
    roles_raw = raw.get("roles", {})
    result: dict[str, ModelSpec] = {}
    for role in REQUIRED_ROLES:
        if role not in roles_raw:
            raise KeyError(f"Missing required role: {role}")
        merged = {**defaults, **roles_raw[role]}
        result[role] = ModelSpec(
            provider=merged["provider"],
            api_base=merged.get("api_base"),
            model=merged["model"],
            temperature=float(merged.get("temperature", 0.1)),
            max_tokens=int(merged["max_tokens"]),
        )
    return result


def load_watched_repos_config(path: Path) -> list[WatchedRepo]:
    raw = yaml.safe_load(path.read_text())
    return [
        WatchedRepo(
            owner=r["owner"],
            name=r["name"],
            default_branch=r.get("default_branch", "main"),
        )
        for r in raw["repos"]
    ]
