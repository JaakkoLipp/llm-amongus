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

    async def decide_move(self, current, options, present, memory, alert=None) -> str:
        # Reactor meltdown: crewmates rush to a fix room.
        if alert and alert.get("kind") == "reactor" and self.role == Role.CREWMATE:
            step = alert.get("step")
            if step and step in options:
                self.last_reasoning = f"Reactor meltdown — rushing toward {step} to fix it."
                return step
            if step is None:  # already at a fix room
                self.last_reasoning = "At the reactor — staying to fix it."
                return "stay"
        # Spread out across the ship (so isolated encounters — and kills — happen);
        # occasionally stay put to do another task.
        if self.rng.random() < 0.8 and options:
            dest = self.rng.choice(options)
            self.last_reasoning = (
                f"Heading to {dest} to find a target alone."
                if self.role == Role.IMPOSTOR
                else f"Moving to {dest} to spread out and do tasks."
            )
            return dest
        self.last_reasoning = "Staying put."
        return "stay"

    async def decide_emergency(self, memory, alive, reason) -> bool:
        # Call a meeting when you actually saw a kill; sometimes after a vent.
        if self._witnessed_killer(memory):
            self.last_reasoning = "I witnessed a murder — calling an emergency meeting now."
            return True
        if any("use a vent" in m for m in memory) and self.rng.random() < 0.5:
            self.last_reasoning = "I saw someone vent — worth calling a meeting."
            return True
        return False

    async def decide_impostor_action(
        self, room, targets, others_here, vent_targets, can_sabotage, memory
    ) -> str:
        # Kill in a true 1-on-1 (no witnesses left behind).
        if targets and len(others_here) == 1:
            self.last_reasoning = f"Alone with {targets[0]} — safe to strike, no witnesses."
            return f"kill {targets[0]}"
        # Witnesses around: occasionally slip away through a vent.
        if others_here and vent_targets and self.rng.random() < 0.4:
            dest = self.rng.choice(vent_targets)
            self.last_reasoning = f"Too crowded — venting to {dest} to escape unseen."
            return f"vent {dest}"
        # Sometimes sabotage: usually lights for cover, occasionally a reactor
        # meltdown to force chaos / a possible sabotage win.
        if can_sabotage and self.rng.random() < 0.25:
            if self.rng.random() < 0.4:
                self.last_reasoning = "Triggering a reactor meltdown to pressure the crew."
                return "sabotage reactor"
            self.last_reasoning = "Cutting the lights to blind the crew."
            return "sabotage lights"
        self.last_reasoning = "Nothing clean here; I'll fake a task."
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
            self.last_reasoning = f"I personally saw {killer} kill someone. Voting {killer}."
            return killer
        # 2) Trust a credible accusation in the discussion.
        m = re.search(r"It's (\w+), vote them out", transcript or "")
        if m and m.group(1) in alive and m.group(1) != self.name:
            self.last_reasoning = f"{m.group(1)} was called out by a witness; I'll back that."
            return m.group(1)
        # 3) Deduce — imperfectly: sometimes suspect someone seen near the death
        #    room. The noise keeps offline games from being a crewmate sweep
        #    (real LLM crewmates are inconsistent too).
        if self.rng.random() < 0.45:
            suspect = self._suspect_from_scene(memory, alive)
            if suspect:
                self.last_reasoning = f"I saw {suspect} near where the body was found."
                return suspect
        others = [p for p in alive if p != self.name]
        if others and self.rng.random() < 0.3:
            pick = self.rng.choice(others)
            self.last_reasoning = f"No hard evidence; {pick} feels off."
            return pick
        self.last_reasoning = "I have no real evidence, so I'll skip."
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
