"""Offline scripted agent — lets a full game run with zero API keys.

It simulates a player of a configurable capability level rather than calling a
model: on tasks it returns the correct answer with probability ``skill`` (so the
evaluation harness still produces a sensible distribution for smoke tests), and
it uses simple rules for movement, kills, discussion, and voting.
"""
from __future__ import annotations

import random

from ..game.models import Role
from ..game.tasks import Task
from .base import Agent

_WRONG = ["42", "potato", "unknown", "0", "banana"]


class HeuristicAgent(Agent):
    def __init__(self, name: str, role: Role, model: str = "heuristic", *, skill: float = 0.7, rng=None):
        super().__init__(name, role, model)
        self.skill = skill
        self.rng = rng or random.Random()
        # Remembered suspicion fed by the engine's observation log.
        self._suspects: list[str] = []

    async def act_task(self, task: Task) -> str:
        if self.rng.random() < self.skill:
            return f"ANSWER: {task.expected}"
        return f"ANSWER: {self.rng.choice(_WRONG)}"

    async def decide_move(self, current: str, options: list[str]) -> str:
        return self.rng.choice(options + ["stay"])

    async def decide_kill(self, targets: list[str], room: str) -> str:
        # Kill only when exactly one witness-free target is present (safer play).
        if targets and self.rng.random() < 0.6:
            return self.rng.choice(targets)
        return "pass"

    async def discuss(self, observation_log: str, transcript: str, alive: list[str]) -> str:
        if self.role == Role.IMPOSTOR:
            return self.rng.choice(
                [
                    "I was doing tasks the whole time, not me.",
                    "We should think carefully before voting anyone out.",
                    f"I saw nothing suspicious in my area.",
                ]
            )
        return self.rng.choice(
            [
                "Where was everyone when it happened?",
                "I completed my tasks and saw no one acting strange near me.",
                "Let's compare who was together.",
            ]
        )

    async def vote(self, observation_log: str, transcript: str, alive: list[str]) -> str:
        others = [p for p in alive if p != self.name]
        # Crewmates lean toward someone named in the observation log; else skip.
        for name in others:
            if name in observation_log and self.rng.random() < 0.5:
                return name
        if others and self.rng.random() < 0.4:
            return self.rng.choice(others)
        return "skip"
