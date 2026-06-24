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


def _reasoning(reply: str, limit: int = 240) -> str:
    """The model's rationale = its reply minus the final `ANSWER:` line."""
    lines = [ln for ln in reply.splitlines() if not ln.strip().upper().startswith("ANSWER")]
    text = " ".join(ln.strip() for ln in lines if ln.strip())
    return text[:limit].strip()


def _parse_action(reply: str, targets: list[str], vent_targets: list[str], can_sabotage: bool) -> str:
    """Normalize an impostor's free-form choice into an action token."""
    ans = extract_answer(reply).lower()
    if can_sabotage and "sabotage" in ans:
        if "reactor" in ans or "meltdown" in ans:
            kind = "reactor"
        elif "comm" in ans:
            kind = "comms"
        else:
            kind = "lights"
        return f"sabotage {kind}"
    if "vent" in ans:
        for r in vent_targets:
            if r.lower() in ans:
                return f"vent {r}"
    for t in targets:  # explicit kill, or just naming a target
        if t.lower() in ans:
            return f"kill {t}"
    return "pass"


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

    async def decide_move(self, current, options, present, memory, alert=None) -> str:
        reply = await self._ask(
            prompts.move_prompt(current, options, present, memory, alert), max_tokens=140
        )
        self.last_reasoning = _reasoning(reply)
        return _match_choice(reply, options, default="stay")

    async def decide_emergency(self, memory, alive, reason) -> bool:
        reply = await self._ask(
            prompts.emergency_prompt(reason, memory, alive), max_tokens=120
        )
        ans = extract_answer(reply).lower()
        called = ans.startswith("y")
        self.last_reasoning = _reasoning(reply) if called else ""
        return called

    async def decide_impostor_action(
        self, room, targets, others_here, vent_targets, can_sabotage, memory
    ) -> str:
        reply = await self._ask(
            prompts.impostor_action_prompt(
                room, targets, others_here, vent_targets, can_sabotage, memory
            ),
            max_tokens=160,
        )
        self.last_reasoning = _reasoning(reply)
        return _parse_action(reply, targets, vent_targets, can_sabotage)

    async def discuss(self, memory, transcript, alive) -> str:
        reply = await self._ask(
            prompts.discussion_prompt(memory, transcript, alive), max_tokens=180
        )
        self.last_reasoning = ""  # the chat message itself is the public statement
        return reply.strip() or "I don't have anything concrete yet."

    async def vote(self, memory, transcript, alive) -> str:
        reply = await self._ask(
            prompts.vote_prompt(memory, transcript, alive), max_tokens=120
        )
        self.last_reasoning = _reasoning(reply)
        return _match_choice(reply, alive + ["skip"], default="skip")
