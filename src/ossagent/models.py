"""Provider-agnostic LLM factory."""

from __future__ import annotations

import os

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from ossagent.config import ModelSpec


def get_llm(role: str, *, config: dict[str, ModelSpec]) -> BaseChatModel:
    spec = config[role]
    if spec.provider in ("moonshot", "openai"):
        api_key_env = {
            "moonshot": "MOONSHOT_API_KEY",
            "openai": "OPENAI_API_KEY",
        }[spec.provider]
        return ChatOpenAI(
            base_url=spec.api_base,
            api_key=os.environ[api_key_env],
            model=spec.model,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
        )
    if spec.provider == "anthropic":
        return ChatAnthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model=spec.model,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
        )
    raise ValueError(f"Unknown provider: {spec.provider}")
