import pytest

from app.config import GameConfig
from app.eval.metrics import Evaluator
from app.game.engine import GameEngine
from app.game.models import EventType


@pytest.mark.asyncio
async def test_full_heuristic_game_terminates_and_scores():
    events = []

    async def emit(ev):
        events.append(ev)

    evaluator = Evaluator()
    cfg = GameConfig(num_players=5, num_impostors=1, max_rounds=8, seed=7,
                     default_model="heuristic")
    engine = GameEngine(cfg, emit=emit, evaluator=evaluator)
    result = await engine.run()

    assert result.winner in ("crewmates", "impostors")
    assert result.rounds >= 1
    # Game start and end are always emitted.
    types = [e.type for e in events]
    assert EventType.GAME_START in types
    assert EventType.GAME_END in types
    # Capability tasks were attempted and recorded.
    board = evaluator.leaderboard()
    assert board, "leaderboard should not be empty"
    assert board[0]["games"] >= 1


@pytest.mark.asyncio
async def test_winner_logic_all_impostors_gone():
    cfg = GameConfig(num_players=4, num_impostors=1, seed=1, default_model="heuristic")
    engine = GameEngine(cfg)
    engine.setup()
    # Kill the impostor off and verify crewmates are declared winners.
    for p in engine.players.values():
        if p.role.value == "impostor":
            p.alive = False
    winner = engine._check_winner()
    assert winner is not None and winner[0] == "crewmates"


@pytest.mark.asyncio
async def test_vote_resolution_ties_and_skips():
    from collections import Counter
    engine = GameEngine(GameConfig(default_model="heuristic"))
    assert engine._resolve_vote(Counter({"Red": 2, "Blue": 2})) is None  # tie
    assert engine._resolve_vote(Counter({"skip": 3, "Red": 1})) is None  # skip plurality
    assert engine._resolve_vote(Counter({"Red": 3, "Blue": 1})) == "Red"


@pytest.mark.asyncio
async def test_mixed_model_specs_are_attributed():
    cfg = GameConfig(
        num_players=4, num_impostors=1, seed=3,
        model_assignments={"Red": "heuristic", "Blue": "heuristic"},
        default_model="heuristic",
    )
    evaluator = Evaluator()
    engine = GameEngine(cfg, evaluator=evaluator)
    await engine.run()
    assert "heuristic" in evaluator.stats
