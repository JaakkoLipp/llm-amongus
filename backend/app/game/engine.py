"""The game engine: a phase state machine that drives agents and emits events.

Design notes
------------
* Every observable happening is a :class:`GameEvent` pushed through an async
  ``emit`` callback. The WebSocket layer streams them to spectators (who are
  omniscient); the headless eval runner ignores them.
* Players have *partial observability with sequential timing*. The action phase
  is processed one player at a time in a shuffled turn order: each acts (moves,
  then does/fakes a task or kills), and everyone co-present records what they
  witness **in real chronological order**, tagged ``R<round>.<tick>``. So a
  crewmate's memory can show "Pink entered Electrical, then Green followed, then
  I saw the body" — the ordering that makes alibis and follow-the-victim reads
  possible.
* A body discovered mid-round ends the action phase immediately and calls a
  meeting (as in Among Us). Agents reason over their own memory, never a global log.
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
        self.tick = 0  # monotonic order index within a round
        self.phase = Phase.LOBBY
        self.players: dict[str, Player] = {}
        self.agents: dict[str, Agent] = {}
        self.tasks: dict[str, list[Task]] = {}
        self.bodies: dict[str, tuple[str, str]] = {}  # victim -> (room, killer)
        self.memory: dict[str, list[str]] = {}        # player -> personal log
        self.sabotage: str | None = None              # "lights" | "comms" | None (this round)
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
        self.memory.setdefault(player, []).append(f"R{self.round}.{self.tick}: {text}")

    def _observe(self, observer: Player, text: str) -> None:
        """Record something a player *witnesses*. Under a 'lights' sabotage,
        crewmates are blind and record nothing they see (impostors still can)."""
        if self.sabotage == "lights" and observer.role == Role.CREWMATE:
            return
        self._remember(observer.name, text)

    async def _emit_thought(self, name: str, action: str) -> None:
        """Stream an agent's private rationale to spectators (omniscient view)."""
        reasoning = (getattr(self.agents[name], "last_reasoning", "") or "").strip()
        if not reasoning:
            return
        await self._emit(
            EventType.THOUGHT, f"{name} (thinking): {reasoning}",
            actor=name, action=action, text=reasoning, private=True,
        )

    def alive(self) -> list[Player]:
        return [p for p in self.players.values() if p.alive]

    def alive_names(self) -> list[str]:
        return [p.name for p in self.alive()]

    def impostors_alive(self) -> list[Player]:
        return [p for p in self.alive() if p.role == Role.IMPOSTOR]

    def crew_alive(self) -> list[Player]:
        return [p for p in self.alive() if p.role == Role.CREWMATE]

    def _occupants(self, room: str, exclude: str | None = None) -> list[Player]:
        return [p for p in self.alive() if p.location == room and p.name != exclude]

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

    # -- per-actor turn steps --------------------------------------------------
    #
    # A round is processed in *ticks*. Each tick selects at most one player per
    # room (so two players sharing a room are sequenced across successive ticks,
    # preserving "follow-the-victim" ordering), then runs those players' turns
    # with overlapping LLM calls: decisions are gathered concurrently, effects are
    # applied in a deterministic order. Players in different rooms can't observe
    # each other mid-turn, so this is safe — and the parallelism grows as the ship
    # disperses. When everyone is crammed in one room it degrades to fully
    # sequential, which is exactly what correctness there requires.

    async def _apply_move(self, actor: Player, choice: str) -> None:
        opts = game_map.neighbors(actor.location)
        if not choice or choice == "stay" or choice not in opts:
            return
        origin = actor.location
        for q in self._occupants(origin, exclude=actor.name):
            self._observe(q, f"saw {actor.name} leave {origin} toward {choice}")
        actor.location = choice
        for q in self._occupants(choice, exclude=actor.name):
            self._observe(q, f"saw {actor.name} arrive in {choice}")
        here = [p.name for p in self._occupants(choice, exclude=actor.name)]
        self._remember(actor.name, f"moved {origin}->{choice}; here now: {', '.join(here) or 'no one'}")
        await self._emit(
            EventType.MOVE, f"{actor.name} moved to {choice}",
            player=actor.name, **{"from": origin, "to": choice},
        )

    async def _commit_task(self, actor: Player, task: Task, raw: str) -> None:
        if not actor.alive:  # killed earlier in this same tick
            return
        att = TaskAttempt(
            player=actor.name, category=task.category, difficulty=task.difficulty,
            prompt=task.prompt, expected=task.expected, answer=extract_answer(raw),
            correct=task.check(raw), round=self.round,
        )
        self.evaluator.record_task(actor.model, att)
        if att.correct:
            actor.tasks_completed += 1
        self._remember(actor.name, f"did a real task in {actor.location}")
        for q in self._occupants(actor.location, exclude=actor.name):
            self._observe(q, f"saw {actor.name} doing a task in {actor.location}")
        await self._emit(
            EventType.TASK_RESULT,
            f"{actor.name} {'completed' if att.correct else 'failed'} a "
            f"{att.category} task ({att.answer!r})",
            player=actor.name, category=att.category, difficulty=att.difficulty,
            correct=att.correct, answer=att.answer, expected=att.expected,
            tasks_completed=actor.tasks_completed, tasks_total=actor.tasks_total,
        )

    async def _fake_task(self, actor: Player) -> None:
        """Impostor pretends to do a task — looks identical to observers."""
        if not actor.alive:
            return
        self._remember(actor.name, f"pretended to do a task in {actor.location}")
        for q in self._occupants(actor.location, exclude=actor.name):
            self._observe(q, f"saw {actor.name} doing a task in {actor.location}")

    async def _commit_kill(self, actor: Player, choice: str) -> bool:
        if actor.kill_cooldown > 0:
            return False
        victim = self.players.get(choice)
        if not (victim and victim.alive and victim.role == Role.CREWMATE
                and victim.location == actor.location):
            return False
        victim.alive = False
        actor.kill_cooldown = self.config.kill_cooldown + 1
        self.bodies[victim.name] = (actor.location, actor.name)
        self.evaluator.record_kill(actor.model)
        witnesses = [p for p in self._occupants(actor.location) if p.name != actor.name]
        for w in witnesses:
            self._observe(w, f"WITNESSED {actor.name} kill {victim.name} in {actor.location}!")
        self._remember(
            actor.name,
            f"killed {victim.name} in {actor.location} "
            f"(witnesses: {', '.join(w.name for w in witnesses) or 'none'})",
        )
        await self._emit(
            EventType.KILL,
            f"{victim.name} was eliminated in {actor.location}"
            + (f" (witnessed by {', '.join(w.name for w in witnesses)})" if witnesses else ""),
            victim=victim.name, room=actor.location, witnesses=[w.name for w in witnesses],
        )
        return True

    async def _commit_vent(self, actor: Player, dest: str) -> bool:
        if dest not in game_map.neighbors(actor.location):
            return False
        origin = actor.location
        # Anyone in the origin catches the vent (a strong tell) — unless blinded.
        for w in self._occupants(origin, exclude=actor.name):
            self._observe(w, f"saw {actor.name} use a vent in {origin}!")
        actor.location = dest  # silent arrival: no destination observations
        self._remember(actor.name, f"vented {origin}->{dest}")
        await self._emit(
            EventType.VENT, f"{actor.name} vented to {dest}",
            player=actor.name, private=True, **{"from": origin, "to": dest},
        )
        return True

    async def _commit_sabotage(self, actor: Player, kind: str) -> None:
        kind = "comms" if "comm" in kind else "lights"
        self.sabotage = kind
        actor.sabotage_cooldown = self.config.sabotage_cooldown + 1
        # Sabotage is public but anonymous — everyone notices, no one sees who.
        for p in self.alive():
            self._remember(p.name, f"the {kind} were sabotaged (by someone unknown)")
        await self._emit(
            EventType.SABOTAGE, f"The {kind} were sabotaged!",
            kind=kind, by=actor.name,  # `by` is spectator-only metadata
        )

    async def _decide_act(self, actor: Player):
        """Make (but don't apply) an actor's post-move action. Returns an intent."""
        if actor.role == Role.CREWMATE:
            if actor.tasks_completed >= actor.tasks_total:
                return ("idle",)
            task = self.tasks[actor.name][actor.tasks_completed]
            raw = await self.agents[actor.name].act_task(task)
            return ("task", task, raw)
        others = self._occupants(actor.location, exclude=actor.name)
        killable = [] if actor.kill_cooldown > 0 else [o.name for o in others if o.role == Role.CREWMATE]
        vent_targets = game_map.neighbors(actor.location)
        can_sabotage = self.sabotage is None and actor.sabotage_cooldown == 0
        token = await self.agents[actor.name].decide_impostor_action(
            actor.location, killable, [o.name for o in others], vent_targets,
            can_sabotage, self.memory[actor.name],
        )
        return ("imp", token)

    async def _apply_act(self, actor: Player, decision: tuple) -> None:
        kind = decision[0]
        if kind == "task":
            await self._commit_task(actor, decision[1], decision[2])
            return
        if kind != "imp":
            return  # "idle": finished crewmate does nothing
        verb, _, arg = decision[1].partition(" ")
        if verb == "kill":
            if not await self._commit_kill(actor, arg.strip()):
                await self._fake_task(actor)
        elif verb == "vent":
            if not await self._commit_vent(actor, arg.strip()):
                await self._fake_task(actor)
        elif verb == "sabotage":
            await self._commit_sabotage(actor, arg.strip())
        else:
            await self._fake_task(actor)

    def _scan_for_report(self) -> tuple[str, str, str] | None:
        """A body is discovered when a living non-killer shares its room.
        Sabotaged comms suppress reports until the round ends."""
        if self.sabotage == "comms":
            return None
        for victim, (room, killer) in list(self.bodies.items()):
            finders = [p for p in self.alive() if p.location == room and p.name != killer]
            if finders:
                reporter = self.rng.choice(finders)
                self._remember(reporter.name, f"found {victim}'s body in {room}")
                return reporter.name, victim, room
        return None

    async def _run_action_phase(self) -> tuple[str, str, str] | None:
        """One round of ticks. Returns a body report the moment one occurs."""
        remaining = self.alive()
        self.rng.shuffle(remaining)

        while remaining:
            # At most one actor per room this tick; the rest wait for a later tick.
            chosen: list[Player] = []
            waiting: list[Player] = []
            rooms_taken: set[str] = set()
            for p in remaining:
                if p.alive and p.location not in rooms_taken:
                    chosen.append(p)
                    rooms_taken.add(p.location)
                elif p.alive:
                    waiting.append(p)
            remaining = waiting
            self.tick += 1

            # MOVE: decide concurrently (overlapping LLM calls), apply in order.
            for a in chosen:
                self.agents[a.name].last_reasoning = ""
            move_choices = await asyncio.gather(*(
                self.agents[a.name].decide_move(
                    a.location, game_map.neighbors(a.location),
                    [o.name for o in self._occupants(a.location, exclude=a.name)],
                    self.memory[a.name],
                )
                for a in chosen
            ))
            for actor, choice in zip(chosen, move_choices):
                await self._apply_move(actor, choice)
                await self._emit_thought(actor.name, "move")

            # ACT: decide concurrently (post-move), apply in order with re-validation.
            actors = [a for a in chosen if a.alive]
            for a in actors:
                self.agents[a.name].last_reasoning = ""
            decisions = await asyncio.gather(*(self._decide_act(a) for a in actors))
            for actor, decision in zip(actors, decisions):
                if not actor.alive:
                    continue
                await self._apply_act(actor, decision)
                await self._emit_thought(actor.name, "act")
                if self._check_winner():
                    return None
                report = self._scan_for_report()
                if report:
                    return report
        return None

    def _decrement_cooldowns(self) -> None:
        for p in self.players.values():
            if p.kill_cooldown > 0:
                p.kill_cooldown -= 1
            if p.sabotage_cooldown > 0:
                p.sabotage_cooldown -= 1

    # -- meeting & voting ------------------------------------------------------

    async def _run_meeting(self, reporter: str, victim: str, room: str) -> None:
        self.phase = Phase.MEETING
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
            await self._emit_thought(name, "vote")

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
            self.tick = 0
            self.sabotage = None  # crew fix any sabotage between rounds
            self.phase = Phase.ACTION
            await self._emit(EventType.PHASE_CHANGE, f"Round {self.round}: action phase")

            report = await self._run_action_phase()
            self._decrement_cooldowns()

            winner = self._check_winner()
            if winner:
                break
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
