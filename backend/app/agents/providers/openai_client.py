"""OpenAI-compatible players: OpenAI, OpenRouter, and Ollama.

All three speak the same Chat Completions wire format, so a single client backed
by the ``openai`` SDK serves them — only the ``base_url`` and API key differ.
"""
from __future__ import annotations

import os

from ...config import PROVIDERS
from .base import ChatMessage, LLMClient, LLMError


class OpenAICompatibleClient(LLMClient):
    def __init__(self, provider: str, model: str):
        super().__init__(provider, model)
        cfg = PROVIDERS[provider]
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMError("openai package not installed. `pip install openai`") from exc

        api_key = os.environ.get(cfg.get("env_key", ""), "")
        if not api_key and cfg.get("key_optional"):
            # Ollama ignores the key but the SDK requires a non-empty string.
            api_key = "ollama"
        if not api_key:
            raise LLMError(
                f"No API key for provider '{provider}'. Set {cfg.get('env_key')}."
            )

        # OpenRouter recommends attribution headers; harmless for the others.
        default_headers = {}
        if provider == "openrouter":
            default_headers = {
                "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://among-llms.local"),
                "X-Title": "Among LLMs",
            }

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=cfg["base_url"],
            default_headers=default_headers or None,
        )

    async def chat(
        self,
        system: str,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 600,
    ) -> str:
        api_messages = [{"role": "system", "content": system}]
        api_messages += [{"role": m.role, "content": m.content} for m in messages]
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # openai raises a family of APIError subclasses
            raise LLMError(f"{self.provider} call failed: {exc}") from exc
        return (resp.choices[0].message.content or "").strip()

    async def aclose(self) -> None:  # pragma: no cover
        await self._client.close()
