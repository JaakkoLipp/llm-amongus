"""Central configuration for Among LLMs.

Everything is overridable via environment variables so the same code runs in a
local dev loop, a CI eval, or a hosted demo.

Players are assigned a *model spec* string of the form ``provider:model``:

    claude:claude-opus-4-8
    openai:gpt-4o
    openrouter:meta-llama/llama-3.1-70b-instruct
    ollama:llama3.1
    heuristic            (offline scripted agent — no API calls, no key needed)

A bare string with no ``provider:`` prefix is treated as a Claude model.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


# Default spec for any LLM player not given an explicit one. Opus 4.8 is the most
# capable widely available Claude model.
DEFAULT_MODEL = os.environ.get("AMONGLLM_DEFAULT_MODEL", "claude:claude-opus-4-8")

# Per-provider connection settings. OpenRouter and Ollama speak the OpenAI wire
# protocol, so they reuse the OpenAI client with a different base URL / key.
PROVIDERS = {
    "claude": {
        "kind": "anthropic",
        "env_key": "ANTHROPIC_API_KEY",
        "alt_env_key": "ANTHROPIC_AUTH_TOKEN",
        "base_url": None,
    },
    "anthropic": {  # alias
        "kind": "anthropic",
        "env_key": "ANTHROPIC_API_KEY",
        "alt_env_key": "ANTHROPIC_AUTH_TOKEN",
        "base_url": None,
    },
    "openai": {
        "kind": "openai",
        "env_key": "OPENAI_API_KEY",
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    },
    "openrouter": {
        "kind": "openai",
        "env_key": "OPENROUTER_API_KEY",
        "base_url": os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    },
    "ollama": {
        "kind": "openai",
        "env_key": "OLLAMA_API_KEY",  # usually unset; Ollama ignores the key
        "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "key_optional": True,
    },
    "heuristic": {"kind": "heuristic"},
}


def split_spec(spec: str) -> tuple[str, str]:
    """``"openai:gpt-4o"`` -> ``("openai", "gpt-4o")``. Bare strings -> claude."""
    spec = spec.strip()
    if spec == "heuristic":
        return "heuristic", "heuristic"
    if ":" in spec:
        provider, _, model = spec.partition(":")
        provider = provider.strip().lower()
        if provider in PROVIDERS:
            return provider, model.strip()
    # No recognised provider prefix -> assume a Claude model id.
    return "claude", spec


def provider_available(provider: str) -> bool:
    """True when the given provider can actually be reached (key present, etc.)."""
    cfg = PROVIDERS.get(provider)
    if not cfg:
        return False
    if cfg["kind"] == "heuristic":
        return True
    if cfg.get("key_optional"):
        return True
    if os.environ.get(cfg.get("env_key", "")):
        return True
    if cfg.get("alt_env_key") and os.environ.get(cfg["alt_env_key"]):
        return True
    return False


def available_providers() -> list[str]:
    """Distinct providers usable right now (deduped, heuristic always last)."""
    seen: list[str] = []
    for name in PROVIDERS:
        if name == "anthropic":  # alias of claude
            continue
        if provider_available(name) and name not in seen:
            seen.append(name)
    return seen


@dataclass
class GameConfig:
    """Tunable parameters for a single match."""

    num_players: int = 5
    num_impostors: int = 1
    max_rounds: int = 8
    # Number of distinct capability tasks a crewmate must complete to "win by tasks".
    tasks_per_crewmate: int = 3
    # How many chat turns each agent gets per meeting before voting.
    discussion_rounds: int = 2
    # Rounds an impostor must wait between kills.
    kill_cooldown: int = 1
    # Rounds an impostor must wait between sabotages.
    sabotage_cooldown: int = 2
    # Seconds to pause between emitted events when streaming a game to spectators.
    # 0 in eval/headless mode; small positive value makes the web UI watchable.
    event_delay: float = float(os.environ.get("AMONGLLM_EVENT_DELAY", "0.0"))
    seed: int | None = None
    # Map of player name -> model spec. Players without an entry use default_model.
    model_assignments: dict[str, str] = field(default_factory=dict)
    default_model: str = DEFAULT_MODEL

    def spec_for(self, player_name: str) -> str:
        return self.model_assignments.get(player_name, self.default_model)
