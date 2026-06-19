from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from model_provider import ProviderConfig, normalize_provider


# Defaults picked so that even a small / medium conversation crosses the
# threshold and forces at least one compaction on the long-context dataset.
DEFAULT_COMPACT_THRESHOLD_TOKENS = 1200
DEFAULT_COMPACT_KEEP_MESSAGES = 6
DEFAULT_TEMPERATURE = 0.0

# Reasonable per-provider model defaults. They can be overridden via env.
_PROVIDER_DEFAULTS = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-1.5-flash",
    "anthropic": "claude-3-5-sonnet-latest",
    "ollama": "llama3.1",
    "openrouter": "openrouter/auto",
}


@dataclass
class LabConfig:
    """Shared configuration for the lab.

    - Paths for repo root, dataset dir, state dir
    - Compact-memory settings (threshold + how many recent messages to keep)
    - Provider settings for the main model and the judge model
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def _load_env() -> None:
    """Best-effort load of .env from repo root; ignore if not present."""
    try:
        load_dotenv(override=False)
    except Exception:
        # python-dotenv not installed yet; env vars still come from the shell
        pass


def _resolve_api_key(provider: str) -> str | None:
    provider = normalize_provider(provider)
    mapping = {
        "openai": "OPENAI_API_KEY",
        "custom": "CUSTOM_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "ollama": None,  # local
        "openrouter": "OPENROUTER_API_KEY",
    }
    env_name = mapping.get(provider)
    if not env_name:
        return None
    return os.environ.get(env_name) or os.environ.get(f"{provider.upper()}_API_KEY")


def _build_provider_config(role: str, fallback_provider: str) -> ProviderConfig:
    """Build a ProviderConfig from env vars, with sensible defaults.

    ``role`` is either ``"model"`` or ``"judge"`` so that two different
    env prefixes (e.g. ``LLM_MODEL`` and ``JUDGE_MODEL``) can be used.
    """
    prefix = "" if role == "model" else "JUDGE_"

    raw_provider = os.environ.get(f"{prefix}LLM_PROVIDER") or (
        "openai" if role == "model" else "anthropic"
    )
    provider = normalize_provider(raw_provider or fallback_provider)

    default_model = _PROVIDER_DEFAULTS.get(provider, "gpt-4o-mini")
    model_name = os.environ.get(f"{prefix}LLM_MODEL") or default_model

    temperature = float(
        os.environ.get(f"{prefix}LLM_TEMPERATURE", str(DEFAULT_TEMPERATURE))
    )

    api_key = _resolve_api_key(provider) or os.environ.get(f"{prefix}LLM_API_KEY")

    base_url = os.environ.get(f"{prefix}LLM_BASE_URL") or os.environ.get(
        f"{prefix}CUSTOM_BASE_URL"
    )

    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a populated LabConfig.

    - Resolves the repo root or defaults to the current file's parent.
    - Loads `.env` if present.
    - Creates `state/` if missing.
    - Chooses sensible defaults for compact memory and the providers.
    """
    _load_env()

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    threshold = int(
        os.environ.get(
            "COMPACT_THRESHOLD_TOKENS", str(DEFAULT_COMPACT_THRESHOLD_TOKENS)
        )
    )
    keep = int(
        os.environ.get("COMPACT_KEEP_MESSAGES", str(DEFAULT_COMPACT_KEEP_MESSAGES))
    )

    model = _build_provider_config("model", "openai")
    judge_model = _build_provider_config("judge", "anthropic")

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=threshold,
        compact_keep_messages=keep,
        model=model,
        judge_model=judge_model,
    )
