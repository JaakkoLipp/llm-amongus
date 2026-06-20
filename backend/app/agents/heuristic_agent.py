"""Offline scripted agent — lets a full game run with zero API keys.

It simulates a player of a configurable capability level rather than calling a
model: on tasks it returns the correct answer with probability ``skill`` (so the
evaluation harness still produces a sensible distribution for smoke tests), and
it uses simple rules over its *personal memory* for movement, kills, discussion,
and voting — including reacting to kills it witnessed.
"""
from __future__ import annotations

import random
import re

from ..game.models import Role
from ..game.tasks import Task
from .base import Agent

_WRONG = ["42", "potato", "unknown", "0", "banana"]
_WITNESS_RE = re.compile(r"WITNESSED (\w+) kill")


class HeuristicAgent(Agent):
    def __init__(self, name: str, role: Role, model: str = "heuristic", *, skill: float = 0.7, rng=None):
        super().__init__(name, role, model)
        self.skill = skill
        self.rng = rng or random.Random()

    @staticmethod
    def _witnessed_killer(memory: list[str]) -> str | None:
        for line in reversed(memory):
            m = _WITNESS_RE.search(line)
            if m:
                return m.group(1)
        return None

    async def act_task(self, task: Task) -> str:
        if self.rng.random() < self.skill:
            return f"ANSWER: {task.expected}"
        return f"ANSWER: {self.rng.choice(_WRONG)}"

    async def decide_move(self, current, options, present, memory) -> str:
        # Spread out across the ship (so isolated encounters — and kills — happen);
        # occasionally stay put to do another task.
        if self.rng.random() < 0.8 and options:
            return self.rng.choice(options)
        return "stay"

    async def decide_kill(self, targets, others_here, room, memory) -> str:
        # Kill only in a true 1-on-1 (no other witnesses left behind).
        if targets and len(others_here) == 1:
            return targets[0]
        return "pass"

    async def discuss(self, memory, transcript, alive) -> str:
        killer = self._witnessed_killer(memory)
        if killer and killer in alive:
            return f"I saw {killer} kill someone! It's {killer}, vote them out."
        if self.role == Role.IMPOSTOR:
            return self.rng.choice(
                [
                    "I was doing tasks the whole time, not me.",
                    "Let's not rush — we don't have solid evidence yet.",
                    "I saw nothing suspicious where I was.",
                ]
            )
        return self.rng.choice(
            [
                "Where was everyone when it happened?",
                "I was doing my tasks and saw no one act strangely near me.",
                "Let's compare who was together.",
            ]
        )

    async def vote(self, memory, transcript, alive) -> str:
        # 1) Caught red-handed beats everything.
        killer = self._witnessed_killer(memory)
        if killer and killer in alive:
            return killer
        # 2) Trust a credible accusation in the discussion.
        m = re.search(r"It's (\w+), vote them out", transcript or "")
        if m and m.group(1) in alive and m.group(1) != self.name:
            return m.group(1)
        # 3) Deduce — imperfectly: sometimes suspect someone seen near the death
        #    room. The noise keeps offline games from being a crewmate sweep
        #    (real LLM crewmates are inconsistent too).
        if self.rng.random() < 0.45:
            suspect = self._suspect_from_scene(memory, alive)
            if suspect:
                return suspect
        others = [p for p in alive if p != self.name]
        if others and self.rng.random() < 0.3:
            return self.rng.choice(others)
        return "skip"

    def _suspect_from_scene(self, memory: list[str], alive: list[str]) -> str | None:
        room = None
        for line in reversed(memory):
            m = re.search(r"found dead in (.+?) \(reported", line)
            if m:
                room = m.group(1)
                break
        if not room:
            return None
        seen: list[str] = []
        for line in memory:
            if room in line:
                for nm in re.findall(r"\b([A-Z][a-z]+)\b", line):
                    if nm in alive and nm != self.name:
                        seen.append(nm)
        # Bias toward the most-recently-seen (often the killer) but not always.
        return self.rng.choice(seen[-3:]) if seen else None
