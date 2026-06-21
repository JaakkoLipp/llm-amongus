"""Core data structures shared across the engine, agents, and API layer."""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Role(str, Enum):
    CREWMATE = "crewmate"
    IMPOSTOR = "impostor"


class Phase(str, Enum):
    LOBBY = "lobby"
    ACTION = "action"      # players move + do tasks; impostors may kill
    MEETING = "meeting"    # live discussion after a body/emergency
    VOTING = "voting"      # players cast ejection votes
    ENDED = "ended"


@dataclass
class TaskAttempt:
    """One capability-test attempt by a player — the unit of capability eval."""

    player: str
    category: str
    difficulty: int
    prompt: str
    expected: str
    answer: str
    correct: bool
    round: int


@dataclass
class Player:
    name: str
    role: Role
    model: str
    location: str
    alive: bool = True
    # Index into the player's personal task list (crewmates only).
    tasks_completed: int = 0
    tasks_total: int = 0
    kill_cooldown: int = 0
    sabotage_cooldown: int = 0
    emergencies_left: int = 0
    ejected: bool = False

    def public_view(self) -> dict[str, Any]:
        """What spectators (and other agents) may see — never leaks role."""
        return {
            "name": self.name,
            "location": self.location,
            "alive": self.alive,
            "ejected": self.ejected,
            "tasks_completed": self.tasks_completed,
            "tasks_total": self.tasks_total,
        }


class EventType(str, Enum):
    GAME_START = "game_start"
    PHASE_CHANGE = "phase_change"
    MOVE = "move"
    TASK_RESULT = "task_result"
    KILL = "kill"
    BODY_REPORTED = "body_reported"
    MEETING_START = "meeting_start"
    CHAT = "chat"
    VOTE = "vote"
    EJECTION = "ejection"
    GAME_END = "game_end"
    INFO = "info"
    THOUGHT = "thought"      # an agent's private reasoning (spectator-only)
    VENT = "vent"            # impostor secret relocation
    SABOTAGE = "sabotage"    # impostor sabotage triggered
    EMERGENCY = "emergency"  # a player called an emergency meeting
    FIX = "fix"              # a crewmate fixed a reactor panel


@dataclass
class GameEvent:
    """Everything that happens is an event; the WS layer streams these verbatim."""

    type: EventType
    round: int
    phase: Phase
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        d["phase"] = self.phase.value
        return d
