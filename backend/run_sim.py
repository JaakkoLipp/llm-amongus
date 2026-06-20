#!/usr/bin/env python3
"""Headless evaluation runner for Among LLMs.

Run a batch of games (optionally with a mix of providers) and print a per-model
capability leaderboard. Works with zero API keys — unconfigured players fall
back to the offline heuristic agent.

Examples
--------
    # 10 all-heuristic games (offline smoke test)
    python run_sim.py --games 10

    # Claude vs OpenAI vs Ollama, 5 games
    python run_sim.py --games 5 \\
        --player Red=claude:claude-opus-4-8 \\
        --player Blue=openai:gpt-4o-mini \\
        --player Green=ollama:llama3.1 \\
        --player Pink=heuristic --player Orange=heuristic
"""
from __future__ import annotations

import argparse
import asyncio
import json

from app.config import GameConfig
from app.eval.metrics import Evaluator
from app.game.engine import GameEngine


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Among LLMs eval runner")
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--players", type=int, default=5, help="player count if not all named")
    ap.add_argument("--impostors", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--tasks", type=int, default=3, help="tasks per crewmate")
    ap.add_argument("--default-model", default="heuristic")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--player", action="append", default=[],
        metavar="NAME=SPEC", help="assign a model spec to a player (repeatable)",
    )
    ap.add_argument("--json", action="store_true", help="print leaderboard as JSON")
    return ap.parse_args()


def build_assignments(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--player expects NAME=SPEC, got {item!r}")
        name, spec = item.split("=", 1)
        out[name.strip()] = spec.strip()
    return out


async def main() -> None:
    args = parse_args()
    assignments = build_assignments(args.player)
    evaluator = Evaluator()
    n_players = max(args.players, len(assignments))

    for g in range(args.games):
        cfg = GameConfig(
            num_players=n_players,
            num_impostors=args.impostors,
            max_rounds=args.rounds,
            tasks_per_crewmate=args.tasks,
            seed=(args.seed + g) if args.seed is not None else None,
            model_assignments=assignments,
            default_model=args.default_model,
        )
        engine = GameEngine(cfg, evaluator=evaluator)
        result = await engine.run()
        print(f"Game {g + 1}: {result.winner} win — {result.reason} ({result.rounds} rounds)")

    board = evaluator.leaderboard()
    if args.json:
        print(json.dumps(board, indent=2))
        return

    print("\n=== Leaderboard (by capability score) ===")
    header = f"{'model':<42}{'score':>7}{'tasks':>8}{'win%':>7}{'vote%':>7}{'games':>7}"
    print(header)
    print("-" * len(header))
    for r in board:
        print(
            f"{r['model']:<42}{r['capability_score']:>7}"
            f"{r['task_accuracy']*100:>7.0f}%{r['win_rate']*100:>6.0f}%"
            f"{r['vote_precision']*100:>6.0f}%{r['games']:>7}"
        )


if __name__ == "__main__":
    asyncio.run(main())
