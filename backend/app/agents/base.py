"""Abstract player agent. Concrete agents are LLM-backed or offline heuristic."""
from __future__ import annotations

import abc

from ..game.models import Role
from ..game.tasks import Task


class Agent(abc.ABC):
    def __init__(self, name: str, role: Role, model: str):
        self.name = name
        self.role = role
        self.model = model  # the model spec, e.g. "openai:gpt-4o"

    @abc.abstractmethod
    async def act_task(self, task: Task) -> str:
        """Return a free-form answer to a capability task."""

    @abc.abstractmethod
    async def decide_move(self, current: str, options: list[str]) -> str:
        """Return a room name from ``options`` or 'stay'."""

    @abc.abstractmethod
    async def decide_kill(self, targets: list[str], room: str) -> str:
        """Impostor only: return a target name or 'pass'."""

    @abc.abstractmethod
    async def discuss(self, observation_log: str, transcript: str, alive: list[str]) -> str:
        """Return one short chat message for the meeting."""

    @abc.abstractmethod
    async def vote(self, observation_log: str, transcript: str, alive: list[str]) -> str:
        """Return the name of a player to eject, or 'skip'."""
