"""Provider-agnostic LLM factory."""

from __future__ import annotations

import os

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from ossagent.config import ModelSpec


def _require_env(name: str, *, role: str, provider: str) -> str:
    try:
        return os.environ[name]
    except KeyError as e:
        raise RuntimeError(
            f"{name} is not set (required for role={role!r}, provider={provider!r})"
        ) from e


def get_llm(role: str, *, config: dict[str, ModelSpec]) -> BaseChatModel:
    """Return a chat model for the given role, looked up from `config`."""
    if role not in config:
        raise ValueError(f"Unknown role: {role!r}. Known roles: {sorted(config)}")
    spec = config[role]
    if spec.provider in ("moonshot", "openai"):
        api_key_env = {
            "moonshot": "MOONSHOT_API_KEY",
            "openai": "OPENAI_API_KEY",
        }[spec.provider]
        return ChatOpenAI(
            base_url=spec.api_base,
            api_key=_require_env(api_key_env, role=role, provider=spec.provider),
            model=spec.model,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
        )
    if spec.provider == "anthropic":
        return ChatAnthropic(
            api_key=_require_env("ANTHROPIC_API_KEY", role=role, provider=spec.provider),
            model=spec.model,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
        )
    raise ValueError(f"Unknown provider: {spec.provider}")
