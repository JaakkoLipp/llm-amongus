"""Evaluation harness: turn games into per-model capability scores.

Two axes are measured:
  * Capability  — task accuracy (raw problem solving under game pressure).
  * Social play — winning, voting correctly, deceiving (impostor), surviving.

Results aggregate by *model spec* so you can run mixed games (e.g. a Claude, an
OpenAI, and an Ollama player at the same table) and compare them directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from ..game.models import Role, TaskAttempt


@dataclass
class ModelStats:
    model: str
    task_attempts: int = 0
    task_correct: int = 0
    task_by_category: dict[str, list[int]] = field(default_factory=dict)  # cat -> [correct, total]
    games: int = 0
    wins: int = 0
    games_as_impostor: int = 0
    wins_as_impostor: int = 0
    games_as_crewmate: int = 0
    wins_as_crewmate: int = 0
    kills: int = 0
    votes_cast: int = 0
    correct_votes: int = 0  # crewmate voted for an actual impostor
    survived: int = 0       # alive (not killed/ejected) at game end

    @property
    def task_accuracy(self) -> float:
        return self.task_correct / self.task_attempts if self.task_attempts else 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def vote_precision(self) -> float:
        return self.correct_votes / self.votes_cast if self.votes_cast else 0.0

    @property
    def deception_rate(self) -> float:
        """How often this model wins when cast as the impostor."""
        return self.wins_as_impostor / self.games_as_impostor if self.games_as_impostor else 0.0

    @property
    def capability_score(self) -> float:
        """A single 0-100 headline number blending the measured axes."""
        return round(
            100
            * (
                0.45 * self.task_accuracy
                + 0.25 * self.win_rate
                + 0.20 * self.vote_precision
                + 0.10 * (self.survived / self.games if self.games else 0.0)
            ),
            1,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(
            task_accuracy=round(self.task_accuracy, 3),
            win_rate=round(self.win_rate, 3),
            vote_precision=round(self.vote_precision, 3),
            deception_rate=round(self.deception_rate, 3),
            capability_score=self.capability_score,
            crew_win_rate=round(self.wins_as_crewmate / self.games_as_crewmate, 3)
            if self.games_as_crewmate else 0.0,
            impostor_win_rate=round(self.deception_rate, 3),
            survival_rate=round(self.survived / self.games, 3) if self.games else 0.0,
            category_accuracy={
                cat: round(c / t, 3) if t else 0.0
                for cat, (c, t) in self.task_by_category.items()
            },
        )
        return d


class Evaluator:
    """Accumulates stats across any number of games (one per process or shared)."""

    def __init__(self) -> None:
        self.stats: dict[str, ModelStats] = {}

    def _get(self, model: str) -> ModelStats:
        if model not in self.stats:
            self.stats[model] = ModelStats(model=model)
        return self.stats[model]

    def record_task(self, model: str, attempt: TaskAttempt) -> None:
        s = self._get(model)
        s.task_attempts += 1
        bucket = s.task_by_category.setdefault(attempt.category, [0, 0])
        bucket[1] += 1
        if attempt.correct:
            s.task_correct += 1
            bucket[0] += 1

    def record_vote(self, model: str, *, correct: bool) -> None:
        s = self._get(model)
        s.votes_cast += 1
        if correct:
            s.correct_votes += 1

    def record_kill(self, model: str) -> None:
        self._get(model).kills += 1

    def record_game(
        self,
        model: str,
        *,
        role: Role,
        won: bool,
        survived: bool,
    ) -> None:
        s = self._get(model)
        s.games += 1
        if won:
            s.wins += 1
        if survived:
            s.survived += 1
        if role == Role.IMPOSTOR:
            s.games_as_impostor += 1
            if won:
                s.wins_as_impostor += 1
        else:
            s.games_as_crewmate += 1
            if won:
                s.wins_as_crewmate += 1

    def leaderboard(self) -> list[dict[str, Any]]:
        rows = [s.to_dict() for s in self.stats.values()]
        rows.sort(key=lambda r: r["capability_score"], reverse=True)
        return rows
