"""An LLM-backed player, usable with any provider via an LLMClient.

Each decision is a single stateless call carrying the player's awareness in the
prompt. That keeps behaviour identical across providers and avoids per-provider
history bookkeeping — important for a fair capability comparison.
"""
from __future__ import annotations

from ..game.models import Role
from ..game.tasks import Task, extract_answer
from . import prompts
from .base import Agent
from .providers.base import ChatMessage, LLMClient, LLMError


def _match_choice(reply: str, options: list[str], default: str) -> str:
    """Map a free-form reply onto one of ``options`` (case-insensitive)."""
    ans = extract_answer(reply).lower().strip(".!? ")
    for opt in options:
        if ans == opt.lower():
            return opt
    for opt in options:  # substring fallback ("move to reactor" -> Reactor)
        if opt.lower() in ans:
            return opt
    return default


class LLMAgent(Agent):
    def __init__(self, name: str, role: Role, model: str, client: LLMClient):
        super().__init__(name, role, model)
        self.client = client
        self.system = prompts.system_prompt(name, role)

    async def _ask(self, user_text: str, *, max_tokens: int = 500) -> str:
        try:
            return await self.client.chat(
                self.system, [ChatMessage("user", user_text)], max_tokens=max_tokens
            )
        except LLMError:
            return ""  # treated as a non-answer / pass by callers

    async def act_task(self, task: Task) -> str:
        return await self._ask(prompts.task_prompt(task.prompt), max_tokens=700)

    async def decide_move(self, current, options, present, memory) -> str:
        reply = await self._ask(
            prompts.move_prompt(current, options, present, memory), max_tokens=140
        )
        return _match_choice(reply, options, default="stay")

    async def decide_kill(self, targets, others_here, room, memory) -> str:
        reply = await self._ask(
            prompts.kill_prompt(targets, others_here, room, memory), max_tokens=140
        )
        return _match_choice(reply, targets, default="pass")

    async def discuss(self, memory, transcript, alive) -> str:
        reply = await self._ask(
            prompts.discussion_prompt(memory, transcript, alive), max_tokens=180
        )
        return reply.strip() or "I don't have anything concrete yet."

    async def vote(self, memory, transcript, alive) -> str:
        reply = await self._ask(
            prompts.vote_prompt(memory, transcript, alive), max_tokens=120
        )
        return _match_choice(reply, alive + ["skip"], default="skip")
