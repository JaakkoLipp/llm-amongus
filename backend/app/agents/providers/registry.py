"""Turn a ``provider:model`` spec into a cached LLM client."""
from __future__ import annotations

from ...config import PROVIDERS, split_spec
from .base import LLMClient, LLMError

# One client per spec, reused across players/games in the process.
_CACHE: dict[str, LLMClient] = {}


def build_client(provider: str, model: str) -> LLMClient:
    cfg = PROVIDERS.get(provider)
    if not cfg:
        raise LLMError(f"Unknown provider '{provider}'")
    kind = cfg["kind"]
    if kind == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(model, provider=provider)
    if kind == "openai":
        from .openai_client import OpenAICompatibleClient

        return OpenAICompatibleClient(provider, model)
    raise LLMError(f"Provider '{provider}' has no chat client (kind={kind})")


def client_for_spec(spec: str) -> LLMClient | None:
    """Return a cached client for the spec, or None for the heuristic agent.

    Raises LLMError if a real provider is requested but cannot be constructed.
    """
    provider, model = split_spec(spec)
    if provider == "heuristic":
        return None
    key = f"{provider}:{model}"
    if key not in _CACHE:
        _CACHE[key] = build_client(provider, model)
    return _CACHE[key]
