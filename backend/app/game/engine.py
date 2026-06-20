"""The game engine: a phase state machine that drives agents and emits events.

Design notes
------------
* Every observable happening is a :class:`GameEvent` pushed through an async
  ``emit`` callback. The WebSocket layer streams them to spectators (who are
  omniscient); the headless eval runner ignores them.
* Players have *partial observability*. Each player keeps a personal memory of
  what it could actually witness — who shared its room, who entered/left, who
  appeared to do tasks, and any kill it saw. Agents reason over their own memory,
  not a global log. This is what makes meetings real: alibis, sightings, and
  caught-in-the-act accusations all come from genuine in-world observation.
* Agent calls within a phase run concurrently with ``asyncio.gather`` so a table
  of slow remote models doesn't serialize into very long rounds.
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
from .models import EventType, GameEvent, Phase, Player, Role, TaskAttempt
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
        self.memory: dict[str, list[str]] = {}  # player -> personal observation log
        self.event_count = 0

    # -- helpers ---------------------------------------------------------------

    async def _emit(self, etype: EventType, message: str, **data) -> None:
        self.event_count += 1
        await self.emit_fn(
            GameEvent(type=etype, round=self.round, phase=self.phase, message=message, data=data)
        )
        if self.config.event_delay:
            await asyncio.sleep(self.config.event_delay)

    def _remember(self, player: str, text: str) -> None:
        self.memory.setdefault(player, []).append(f"R{self.round}: {text}")

    def alive(self) -> list[Player]:
        return [p for p in self.players.values() if p.alive]

    def alive_names(self) -> list[str]:
        return [p.name for p in self.alive()]

    def impostors_alive(self) -> list[Player]:
        return [p for p in self.alive() if p.role == Role.IMPOSTOR]

    def crew_alive(self) -> list[Player]:
        return [p for p in self.alive() if p.role == Role.CREWMATE]

    def _present_with(self, player: Player, locations: dict[str, str]) -> list[str]:
        """Living players (other than ``player``) sharing its room in ``locations``."""
        return [
            q.name for q in self.alive()
            if q.name != player.name and locations.get(q.name) == locations.get(player.name)
        ]

    # -- setup -----------------------------------------------------------------

    def setup(self) -> None:
        n = self.config.num_players
        names = PLAYER_NAMES[:n]
        impostor_names = set(self.rng.sample(names, self.config.num_impostors))
        bank = TaskBank(self.rng)
        for name in names:
            role = Role.IMPOSTOR if name in impostor_names else Role.CREWMATE
            spec = self.config.spec_for(name)
            player = Player(name=name, role=role, model=spec, location=game_map.START_ROOM)
            if role == Role.CREWMATE:
                tasks = bank.draw(self.config.tasks_per_crewmate)
                self.tasks[name] = tasks
                player.tasks_total = len(tasks)
            self.players[name] = player
            self.agents[name] = make_agent(name, role, spec, rng=self.rng)
            self.memory[name] = []

    # -- action phase ----------------------------------------------------------

    async def _run_tasks(self, start_rooms: dict[str, str]) -> None:
        """Crewmates attempt real tasks; impostors fake them. Co-located players
        witness who *appeared* to do a task (real or faked are indistinguishable)."""
        living = self.alive()
        real_actors = [
            p for p in living
            if p.role == Role.CREWMATE and p.tasks_completed < p.tasks_total
        ]
        fake_actors = [p for p in living if p.role == Role.IMPOSTOR]

        async def attempt(p: Player) -> tuple[Player, TaskAttempt]:
            task = self.tasks[p.name][p.tasks_completed]
            raw = await self.agents[p.name].act_task(task)
            correct = task.check(raw)
            return p, TaskAttempt(
                player=p.name, category=task.category, difficulty=task.difficulty,
                prompt=task.prompt, expected=task.expected, answer=extract_answer(raw),
                correct=correct, round=self.round,
            )

        for p, att in await asyncio.gather(*(attempt(p) for p in real_actors)):
            self.evaluator.record_task(p.model, att)
            if att.correct:
                p.tasks_completed += 1
            self._remember(p.name, f"did a real task in {start_rooms[p.name]}")
            await self._emit(
                EventType.TASK_RESULT,
                f"{p.name} {'completed' if att.correct else 'failed'} a "
                f"{att.category} task ({att.answer!r})",
                player=p.name, category=att.category, difficulty=att.difficulty,
                correct=att.correct, answer=att.answer, expected=att.expected,
                tasks_completed=p.tasks_completed, tasks_total=p.tasks_total,
            )

        for p in fake_actors:
            self._remember(p.name, f"pretended to do a task in {start_rooms[p.name]}")

        # Witnessing: anyone sharing the actor's room sees the (apparent) task.
        for actor in real_actors + fake_actors:
            room = start_rooms[actor.name]
            for obs in living:
                if obs.name != actor.name and start_rooms[obs.name] == room:
                    self._remember(obs.name, f"saw {actor.name} doing a task in {room}")

    async def _run_moves(self, start_rooms: dict[str, str]) -> None:
        movers = self.alive()

        async def pick(p: Player) -> tuple[Player, str]:
            opts = game_map.neighbors(p.location)
            present = self._present_with(p, start_rooms)
            choice = await self.agents[p.name].decide_move(
                p.location, opts, present, self.memory[p.name]
            )
            return p, choice

        moves: dict[str, tuple[str, str]] = {}  # name -> (from, to)
        for p, choice in await asyncio.gather(*(pick(p) for p in movers)):
            if choice and choice != "stay" and choice in game_map.neighbors(p.location):
                old = p.location
                p.location = choice
                moves[p.name] = (old, choice)

        end_rooms = {p.name: p.location for p in self.alive()}

        for name, (old, new) in moves.items():
            self._remember(name, f"moved from {old} to {new}")
            # Players left behind in the origin room see the departure.
            for q in self.alive():
                if q.name != name and start_rooms.get(q.name) == old:
                    self._remember(q.name, f"saw {name} leave {old} toward {new}")
            # Players already in / arriving to the destination see the arrival.
            for q in self.alive():
                if q.name != name and end_rooms.get(q.name) == new:
                    self._remember(q.name, f"saw {name} arrive in {new}")
            await self._emit(
                EventType.MOVE, f"{name} moved to {new}",
                player=name, **{"from": old, "to": new},
            )

    async def _run_kills(self) -> None:
        for imp in self.impostors_alive():
            if imp.kill_cooldown > 0:
                continue
            room_occupants = [p for p in self.alive() if p.location == imp.location and p.name != imp.name]
            targets = [p for p in room_occupants if p.role == Role.CREWMATE]
            if not targets:
                continue
            others_here = [p.name for p in room_occupants]  # everyone but victim witnesses
            target_name = await self.agents[imp.name].decide_kill(
                [p.name for p in targets], others_here, imp.location, self.memory[imp.name]
            )
            victim = self.players.get(target_name)
            if not (victim and victim.alive and victim.role == Role.CREWMATE
                    and victim.location == imp.location):
                continue

            victim.alive = False
            imp.kill_cooldown = self.config.kill_cooldown + 1
            self.bodies[victim.name] = victim.location
            self.evaluator.record_kill(imp.model)
            # Anyone else in the room witnessed the murder — a confirmed sighting.
            actual_witnesses = [
                p for p in self.alive()
                if p.location == imp.location and p.name not in (imp.name, victim.name)
            ]
            for w in actual_witnesses:
                self._remember(w.name, f"WITNESSED {imp.name} kill {victim.name} in {imp.location}!")
            self._remember(
                imp.name,
                f"killed {victim.name} in {imp.location} "
                f"(witnesses: {', '.join(w.name for w in actual_witnesses) or 'none'})",
            )
            await self._emit(
                EventType.KILL,
                f"{victim.name} was eliminated in {victim.location}"
                + (f" (witnessed by {', '.join(w.name for w in actual_witnesses)})" if actual_witnesses else ""),
                victim=victim.name, room=victim.location,
                witnesses=[w.name for w in actual_witnesses],
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
                self._remember(reporter.name, f"found {victim}'s body in {room}")
                return reporter.name, victim, room
        return None

    # -- meeting & voting ------------------------------------------------------

    async def _run_meeting(self, reporter: str, victim: str, room: str) -> None:
        self.phase = Phase.MEETING
        # Everyone learns the public fact of the meeting.
        for p in self.alive():
            self._remember(p.name, f"meeting called: {victim} found dead in {room} (reported by {reporter})")
        self.bodies.clear()
        await self._emit(
            EventType.BODY_REPORTED, f"{reporter} reported {victim}'s body in {room}",
            reporter=reporter, victim=victim, room=room,
        )
        await self._emit(EventType.MEETING_START, "Emergency meeting started")

        transcript: list[str] = []
        for _ in range(self.config.discussion_rounds):
            for name in self.alive_names():
                msg = await self.agents[name].discuss(
                    self.memory[name], "\n".join(transcript), self.alive_names()
                )
                line = f"{name}: {msg}"
                transcript.append(line)
                self._remember(name, f"said in meeting: {msg}")
                await self._emit(EventType.CHAT, line, speaker=name, text=msg)

        await self._run_vote(transcript)

    async def _run_vote(self, transcript: list[str]) -> None:
        self.phase = Phase.VOTING
        alive = self.alive_names()
        tally: Counter[str] = Counter()

        async def cast(name: str) -> tuple[str, str]:
            choice = await self.agents[name].vote(
                self.memory[name], "\n".join(transcript), alive
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

        ejected = self._resolve_vote(tally)
        if ejected:
            p = self.players[ejected]
            p.alive = False
            p.ejected = True
            for q in self.alive():
                self._remember(q.name, f"{ejected} was ejected (was a {p.role.value})")
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
        contenders = [c for c, n in tally.items() if n == count and c != "skip"]
        return contenders[0] if len(contenders) == 1 else None

    # -- win conditions --------------------------------------------------------

    def _check_winner(self) -> tuple[str, str] | None:
        if all(p.tasks_completed >= p.tasks_total
               for p in self.players.values() if p.role == Role.CREWMATE):
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
            players=[{"name": p.name, "model": p.model} for p in self.players.values()],
            map=game_map.ROOMS,
        )

        winner: tuple[str, str] | None = None
        while self.round < self.config.max_rounds and winner is None:
            self.round += 1
            self.phase = Phase.ACTION
            await self._emit(EventType.PHASE_CHANGE, f"Round {self.round}: action phase")

            start_rooms = {p.name: p.location for p in self.alive()}
            for p in self.alive():
                present = self._present_with(p, start_rooms)
                self._remember(
                    p.name,
                    f"in {p.location} with {', '.join(present) if present else 'no one'}",
                )

            await self._run_tasks(start_rooms)
            winner = self._check_winner()
            if winner:
                break

            await self._run_moves(start_rooms)
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
            winner=winner[0], reason=winner[1], rounds=self.round, events=self.event_count,
            players=[
                {
                    "name": p.name, "role": p.role.value, "model": p.model,
                    "alive": p.alive, "ejected": p.ejected,
                    "tasks_completed": p.tasks_completed, "tasks_total": p.tasks_total,
                }
                for p in self.players.values()
            ],
        )
