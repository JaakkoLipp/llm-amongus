"""Abstract player agent. Concrete agents are LLM-backed or offline heuristic.

Decision methods receive the agent's *personal* awareness: who shares its room
(``present``/``witnesses``) and its accumulated observation log (``memory``,
newest last). The engine builds and maintains that memory from what each player
could actually witness.
"""
from __future__ import annotations

import abc

from ..game.models import Role
from ..game.tasks import Task


class Agent(abc.ABC):
    def __init__(self, name: str, role: Role, model: str):
        self.name = name
        self.role = role
        self.model = model  # the model spec, e.g. "openai:gpt-4o"
        # Short private rationale for the most recent decision; the engine reads
        # this after each call and streams it to spectators as a THOUGHT event.
        self.last_reasoning: str = ""

    @abc.abstractmethod
    async def act_task(self, task: Task) -> str:
        """Return a free-form answer to a capability task (solo)."""

    @abc.abstractmethod
    async def decide_move(
        self, current: str, options: list[str], present: list[str], memory: list[str]
    ) -> str:
        """Return a room from ``options`` or 'stay', aware of who is in the room."""

    @abc.abstractmethod
    async def decide_impostor_action(
        self,
        room: str,
        targets: list[str],
        others_here: list[str],
        vent_targets: list[str],
        can_sabotage: bool,
        memory: list[str],
    ) -> str:
        """Impostor's action token, one of:

        * ``"kill <name>"``      — eliminate a co-located crewmate (witnesses expose you)
        * ``"vent <room>"``      — relocate to a neighbour secretly (caught if seen)
        * ``"sabotage lights"``  — blind crewmates for the rest of the round
        * ``"sabotage comms"``   — block body reports / meetings for the rest of the round
        * ``"pass"``             — fake a task to blend in
        """

    @abc.abstractmethod
    async def discuss(self, memory: list[str], transcript: str, alive: list[str]) -> str:
        """Return one short chat message, reasoning over personal memory."""

    @abc.abstractmethod
    async def vote(self, memory: list[str], transcript: str, alive: list[str]) -> str:
        """Return the name of a player to eject, or 'skip'."""
