from __future__ import annotations

from dataclasses import dataclass


# Aliases thường gặp trong env vars / typo
_PROVIDER_ALIASES = {
    "anthorpic": "anthropic",
    "antheropic": "anthropic",
    "claude": "anthropic",
    "chatgpt": "openai",
    "gpt": "openai",
    "google": "gemini",
    "googlegenai": "gemini",
    "lmstudio": "custom",
    "vllm": "custom",
    "openai-compatible": "custom",
    "or": "openrouter",
}


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Map aliases / typos -> canonical provider name.

    - strips whitespace
    - lowercases
    - returns the canonical name from a small alias table
    - unknown provider is returned lowercased (caller can decide)
    """
    if value is None:
        return ""
    cleaned = value.strip().lower().replace("_", "-").replace(" ", "-")
    if cleaned in _PROVIDER_ALIASES:
        return _PROVIDER_ALIASES[cleaned]
    return cleaned


def build_chat_model(config: ProviderConfig):
    """Instantiate a chat model for the selected provider.

    Returns ``None`` if the required SDK is missing, so the agent can fall
    back to deterministic offline mode. This keeps the lab runnable on
    machines without API keys while still supporting every provider.
    """
    provider = normalize_provider(config.provider)
    temperature = config.temperature
    api_key = config.api_key
    base_url = config.base_url
    model_name = config.model_name

    if provider in {"openai", "custom"}:
        try:
            from langchain_openai import ChatOpenAI
        except Exception:
            return None
        kwargs = {"model": model_name, "temperature": temperature}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        # `custom` defaults to OpenAI-compatible base URL
        if provider == "custom" and not base_url:
            kwargs["base_url"] = "http://localhost:1234/v1"
        try:
            return ChatOpenAI(**kwargs)
        except Exception:
            return None

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except Exception:
            return None
        kwargs = {"model": model_name, "temperature": temperature}
        if api_key:
            kwargs["google_api_key"] = api_key
        try:
            return ChatGoogleGenerativeAI(**kwargs)
        except Exception:
            return None

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except Exception:
            return None
        kwargs = {"model": model_name, "temperature": temperature}
        if api_key:
            kwargs["api_key"] = api_key
        try:
            return ChatAnthropic(**kwargs)
        except Exception:
            return None

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except Exception:
            return None
        kwargs = {"model": model_name, "temperature": temperature}
        if base_url:
            kwargs["base_url"] = base_url
        try:
            return ChatOllama(**kwargs)
        except Exception:
            return None

    if provider == "openrouter":
        try:
            from langchain_openrouter import ChatOpenRouter
        except Exception:
            return None
        kwargs = {"model": model_name, "temperature": temperature}
        if api_key:
            kwargs["api_key"] = api_key
        try:
            return ChatOpenRouter(**kwargs)
        except Exception:
            return None

    return None
