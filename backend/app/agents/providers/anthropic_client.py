"""Claude players, via the official Anthropic SDK."""
from __future__ import annotations

from .base import ChatMessage, LLMClient, LLMError


class AnthropicClient(LLMClient):
    def __init__(self, model: str, *, provider: str = "claude"):
        super().__init__(provider, model)
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "anthropic package not installed. `pip install anthropic`"
            ) from exc
        # Resolves credentials from ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN.
        self._client = anthropic.AsyncAnthropic()

    async def chat(
        self,
        system: str,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 600,
    ) -> str:
        import anthropic

        api_messages = [{"role": m.role, "content": m.content} for m in messages]
        try:
            resp = await self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=api_messages,
            )
        except anthropic.APIError as exc:
            raise LLMError(f"Anthropic call failed: {exc}") from exc
        # Concatenate any text blocks; ignore thinking/tool blocks.
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()

    async def aclose(self) -> None:  # pragma: no cover
        await self._client.close()
