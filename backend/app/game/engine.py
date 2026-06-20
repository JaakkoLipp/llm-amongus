"""The game engine: a phase state machine that drives agents and emits events.

Design notes
------------
* Every observable happening is a :class:`GameEvent` pushed through an async
  ``emit`` callback. The WebSocket layer streams them to spectators; the headless
  eval runner ignores them. The engine itself never touches I/O beyond ``emit``.
* Movement is fully public (a shared observation log) so crewmates have alibi
  data to reason over — this keeps the deduction tractable for v1.
* Agent calls within a phase are issued concurrently with ``asyncio.gather`` so a
  table of slow remote models doesn't serialize into very long rounds.
"""
from __future__ import annotations

import asyncio
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from ..agents import Agent, make_agent
from ..config import GameConfig
from ..eval.metrics import Evaluator
from . import map as game_map
from .models import (
    EventType,
    GameEvent,
    Phase,
    Player,
    Role,
    TaskAttempt,
)
from .tasks import Task, TaskBank, extract_answer

PLAYER_NAMES = [
    "Red", "Blue", "Green", "Pink", "Orange",
    "Yellow", "Cyan", "Lime", "White", "Black",
]

EmitFn = Callable[[GameEvent], Awaitable[None]]


@dataclass
class GameResult:
    winner: str  # "crewmates" | "impostors"
    reason: str
    rounds: int
    players: list[dict]
    events: int = 0
    per_model: dict = field(default_factory=dict)


async def _noop_emit(_: GameEvent) -> None:
    return None


class GameEngine:
    def __init__(
        self,
        config: GameConfig | None = None,
        emit: EmitFn | None = None,
        evaluator: Evaluator | None = None,
    ):
        self.config = config or GameConfig()
        self.emit_fn = emit or _noop_emit
        self.evaluator = evaluator or Evaluator()
        self.rng = random.Random(self.config.seed)
        self.round = 0
        self.phase = Phase.LOBBY
        self.players: dict[str, Player] = {}
        self.agents: dict[str, Agent] = {}
        self.tasks: dict[str, list[Task]] = {}
        self.bodies: dict[str, str] = {}  # victim name -> room, awaiting report
        self.round_obs: list[str] = []
        self.event_count = 0

    # -- helpers ---------------------------------------------------------------

    async def _emit(self, etype: EventType, message: str, **data) -> None:
        self.event_count += 1
        await self.emit_fn(
            GameEvent(type=etype, round=self.round, phase=self.phase, message=message, data=data)
        )
        if self.config.event_delay:
            await asyncio.sleep(self.config.event_delay)

    def alive(self) -> list[Player]:
        return [p for p in self.players.values() if p.alive]

    def alive_names(self) -> list[str]:
        return [p.name for p in self.alive()]

    def impostors_alive(self) -> list[Player]:
        return [p for p in self.alive() if p.role == Role.IMPOSTOR]

    def crew_alive(self) -> list[Player]:
        return [p for p in self.alive() if p.role == Role.CREWMATE]

    def obs_text(self) -> str:
        return "\n".join(self.round_obs) if self.round_obs else "(quiet round, nothing notable)"

    # -- setup -----------------------------------------------------------------

    def setup(self) -> None:
        n = self.config.num_players
        names = PLAYER_NAMES[:n]
        impostor_names = set(self.rng.sample(names, self.config.num_impostors))
        bank = TaskBank(self.rng)
        for name in names:
            role = Role.IMPOSTOR if name in impostor_names else Role.CREWMATE
            spec = self.config.spec_for(name)
            player = Player(
                name=name, role=role, model=spec, location=game_map.START_ROOM
            )
            if role == Role.CREWMATE:
                tasks = bank.draw(self.config.tasks_per_crewmate)
                self.tasks[name] = tasks
                player.tasks_total = len(tasks)
            self.players[name] = player
            self.agents[name] = make_agent(name, role, spec, rng=self.rng)

    # -- phases ----------------------------------------------------------------

    async def _run_tasks(self) -> None:
        crew = [p for p in self.crew_alive() if p.tasks_completed < p.tasks_total]
        if not crew:
            return

        async def attempt(p: Player) -> tuple[Player, TaskAttempt]:
            task = self.tasks[p.name][p.tasks_completed]
            raw = await self.agents[p.name].act_task(task)
            correct = task.check(raw)
            att = TaskAttempt(
                player=p.name,
                category=task.category,
                difficulty=task.difficulty,
                prompt=task.prompt,
                expected=task.expected,
                answer=extract_answer(raw),
                correct=correct,
                round=self.round,
            )
            return p, att

        for p, att in await asyncio.gather(*(attempt(p) for p in crew)):
            self.evaluator.record_task(p.model, att)
            if att.correct:
                p.tasks_completed += 1
            await self._emit(
                EventType.TASK_RESULT,
                f"{p.name} {'completed' if att.correct else 'failed'} a "
                f"{att.category} task ({att.answer!r})",
                player=p.name,
                category=att.category,
                difficulty=att.difficulty,
                correct=att.correct,
                answer=att.answer,
                expected=att.expected,
                tasks_completed=p.tasks_completed,
                tasks_total=p.tasks_total,
            )

    async def _run_moves(self) -> None:
        movers = self.alive()

        async def pick(p: Player) -> tuple[Player, str]:
            opts = game_map.neighbors(p.location)
            choice = await self.agents[p.name].decide_move(p.location, opts)
            return p, choice

        for p, choice in await asyncio.gather(*(pick(p) for p in movers)):
            if choice and choice != "stay" and choice in game_map.neighbors(p.location):
                old = p.location
                p.location = choice
                self.round_obs.append(f"{p.name} moved from {old} to {choice}.")
                await self._emit(
                    EventType.MOVE, f"{p.name} moved to {choice}",
                    player=p.name, **{"from": old, "to": choice},
                )

    async def _run_kills(self) -> None:
        for imp in self.impostors_alive():
            if imp.kill_cooldown > 0:
                continue
            here = [
                p for p in self.alive()
                if p.location == imp.location and p.role == Role.CREWMATE
            ]
            if not here:
                continue
            target_name = await self.agents[imp.name].decide_kill(
                [p.name for p in here], imp.location
            )
            victim = self.players.get(target_name)
            if victim and victim.alive and victim.role == Role.CREWMATE and victim.location == imp.location:
                victim.alive = False
                imp.kill_cooldown = self.config.kill_cooldown + 1
                self.bodies[victim.name] = victim.location
                self.evaluator.record_kill(imp.model)
                await self._emit(
                    EventType.KILL,
                    f"{victim.name} was eliminated in {victim.location}",
                    victim=victim.name, room=victim.location,
                )

    def _decrement_cooldowns(self) -> None:
        for p in self.players.values():
            if p.kill_cooldown > 0:
                p.kill_cooldown -= 1

    def _find_body_report(self) -> tuple[str, str, str] | None:
        """Return (reporter, victim, room) if a living player shares a body's room."""
        for victim, room in list(self.bodies.items()):
            witnesses = [p for p in self.alive() if p.location == room]
            if witnesses:
                reporter = self.rng.choice(witnesses)
                return reporter.name, victim, room
        return None

    async def _run_meeting(self, reporter: str, victim: str, room: str) -> None:
        self.phase = Phase.MEETING
        self.round_obs.append(f"{reporter} reported {victim}'s body in {room}.")
        await self._emit(
            EventType.BODY_REPORTED,
            f"{reporter} reported {victim}'s body in {room}",
            reporter=reporter, victim=victim, room=room,
        )
        self.bodies.clear()
        await self._emit(EventType.MEETING_START, "Emergency meeting started")

        transcript: list[str] = []
        alive = self.alive_names()
        for _ in range(self.config.discussion_rounds):
            for name in alive:
                if not self.players[name].alive:
                    continue
                msg = await self.agents[name].discuss(
                    self.obs_text(), "\n".join(transcript), self.alive_names()
                )
                line = f"{name}: {msg}"
                transcript.append(line)
                await self._emit(EventType.CHAT, line, speaker=name, text=msg)

        await self._run_vote(transcript)

    async def _run_vote(self, transcript: list[str]) -> None:
        self.phase = Phase.VOTING
        alive = self.alive_names()
        tally: Counter[str] = Counter()

        async def cast(name: str) -> tuple[str, str]:
            choice = await self.agents[name].vote(
                self.obs_text(), "\n".join(transcript), alive
            )
            return name, choice

        for name, choice in await asyncio.gather(*(cast(n) for n in alive)):
            voter = self.players[name]
            valid = choice if (choice in alive or choice == "skip") else "skip"
            tally[valid] += 1
            if voter.role == Role.CREWMATE and valid != "skip":
                target = self.players.get(valid)
                self.evaluator.record_vote(
                    voter.model, correct=bool(target and target.role == Role.IMPOSTOR)
                )
            await self._emit(EventType.VOTE, f"{name} voted for {valid}", voter=name, choice=valid)

        # Resolve: strict plurality, ties or skip-plurality => no ejection.
        ejected = self._resolve_vote(tally)
        if ejected:
            p = self.players[ejected]
            p.alive = False
            p.ejected = True
            await self._emit(
                EventType.EJECTION,
                f"{ejected} was ejected. They were a {p.role.value}.",
                player=ejected, role=p.role.value, tally=dict(tally),
            )
        else:
            await self._emit(
                EventType.EJECTION, "No one was ejected (tie or skip).",
                player=None, tally=dict(tally),
            )

    def _resolve_vote(self, tally: Counter[str]) -> str | None:
        if not tally:
            return None
        top, count = tally.most_common(1)[0]
        if top == "skip":
            return None
        # Tie among non-skip candidates -> no ejection.
        contenders = [c for c, n in tally.items() if n == count and c != "skip"]
        if len(contenders) != 1:
            return None
        return contenders[0]

    # -- win conditions --------------------------------------------------------

    def _check_winner(self) -> tuple[str, str] | None:
        if all(p.tasks_completed >= p.tasks_total for p in self.players.values() if p.role == Role.CREWMATE):
            return "crewmates", "all tasks completed"
        imp = self.impostors_alive()
        crew = self.crew_alive()
        if not imp:
            return "crewmates", "all impostors ejected"
        if len(imp) >= len(crew):
            return "impostors", "impostors reached parity"
        return None

    # -- main loop -------------------------------------------------------------

    async def run(self) -> GameResult:
        self.setup()
        self.phase = Phase.ACTION
        await self._emit(
            EventType.GAME_START,
            f"Game started with {self.config.num_players} players, "
            f"{self.config.num_impostors} impostor(s)",
            players=[
                {"name": p.name, "model": p.model} for p in self.players.values()
            ],
            map=game_map.ROOMS,
        )

        winner: tuple[str, str] | None = None
        while self.round < self.config.max_rounds and winner is None:
            self.round += 1
            self.phase = Phase.ACTION
            self.round_obs = []
            await self._emit(EventType.PHASE_CHANGE, f"Round {self.round}: action phase")

            await self._run_tasks()
            winner = self._check_winner()
            if winner:
                break

            await self._run_moves()
            await self._run_kills()
            self._decrement_cooldowns()

            winner = self._check_winner()
            if winner:
                break

            report = self._find_body_report()
            if report:
                await self._run_meeting(*report)
                winner = self._check_winner()

        if winner is None:
            winner = ("crewmates", "round limit reached")

        await self._finish(winner)
        return self._build_result(winner)

    async def _finish(self, winner: tuple[str, str]) -> None:
        self.phase = Phase.ENDED
        win_side = winner[0]
        for p in self.players.values():
            won = (
                (win_side == "impostors" and p.role == Role.IMPOSTOR)
                or (win_side == "crewmates" and p.role == Role.CREWMATE)
            )
            survived = p.alive and not p.ejected
            self.evaluator.record_game(p.model, role=p.role, won=won, survived=survived)
        await self._emit(
            EventType.GAME_END,
            f"{win_side.capitalize()} win — {winner[1]}",
            winner=win_side, reason=winner[1],
            reveal=[
                {"name": p.name, "role": p.role.value, "model": p.model, "alive": p.alive}
                for p in self.players.values()
            ],
        )

    def _build_result(self, winner: tuple[str, str]) -> GameResult:
        return GameResult(
            winner=winner[0],
            reason=winner[1],
            rounds=self.round,
            events=self.event_count,
            players=[
                {
                    "name": p.name,
                    "role": p.role.value,
                    "model": p.model,
                    "alive": p.alive,
                    "ejected": p.ejected,
                    "tasks_completed": p.tasks_completed,
                    "tasks_total": p.tasks_total,
                }
                for p in self.players.values()
            ],
        )
