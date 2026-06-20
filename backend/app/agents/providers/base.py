"""Provider-agnostic chat interface used by LLM agents.

Every concrete client exposes one coroutine, ``chat()``, that takes a system
prompt plus an alternating message history and returns the assistant's text.
This keeps the game engine and agent logic completely decoupled from which
vendor (Anthropic, OpenAI, OpenRouter, Ollama) is actually serving a player.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass


class LLMError(RuntimeError):
    """Raised when a provider call fails after retries."""


@dataclass
class ChatMessage:
    role: str  # "user" | "assistant"
    content: str


class LLMClient(abc.ABC):
    """Minimal async chat client. One instance is shared per (provider, model)."""

    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model

    @property
    def spec(self) -> str:
        return f"{self.provider}:{self.model}"

    @abc.abstractmethod
    async def chat(
        self,
        system: str,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 600,
    ) -> str:
        """Return the assistant's reply text for the given conversation."""

    async def aclose(self) -> None:  # pragma: no cover - optional cleanup
        pass
