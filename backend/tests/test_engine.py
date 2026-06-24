import pytest

from app.config import GameConfig
from app.eval.metrics import Evaluator
from app.game import map as gmap
from app.game.engine import GameEngine
from app.game.models import EventType, Role


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
async def test_thoughts_are_streamed():
    events = []

    async def emit(ev):
        events.append(ev)

    engine = GameEngine(GameConfig(num_players=5, num_impostors=1, seed=7,
                                   default_model="heuristic"), emit=emit)
    await engine.run()
    thoughts = [e for e in events if e.type == EventType.THOUGHT]
    assert thoughts, "agents should emit private reasoning"
    assert all(t.data.get("text") for t in thoughts)


@pytest.mark.asyncio
async def test_lights_sabotage_blinds_crewmates_only():
    engine = GameEngine(GameConfig(seed=1, default_model="heuristic"))
    engine.setup()
    engine.round, engine.tick = 1, 1
    crew = next(p for p in engine.players.values() if p.role == Role.CREWMATE)
    imp = next(p for p in engine.players.values() if p.role == Role.IMPOSTOR)
    engine.sabotage = "lights"
    engine._observe(crew, "saw a clue")    # crewmate is blind -> dropped
    engine._observe(imp, "saw a clue")     # impostor still sees
    assert all("saw a clue" not in m for m in engine.memory[crew.name])
    assert any("saw a clue" in m for m in engine.memory[imp.name])
    engine.sabotage = None
    engine._observe(crew, "lights back")   # restored after fix
    assert any("lights back" in m for m in engine.memory[crew.name])


@pytest.mark.asyncio
async def test_vent_relocates_secretly():
    engine = GameEngine(GameConfig(seed=1, default_model="heuristic"))
    engine.setup()
    engine.round, engine.tick = 1, 1
    imp = next(p for p in engine.players.values() if p.role == Role.IMPOSTOR)
    dest = gmap.neighbors(imp.location)[0]
    # Move everyone else away so the destination has no witnesses.
    for p in engine.players.values():
        if p.name != imp.name:
            p.location = "Storage" if dest != "Storage" else "MedBay"
    ok = await engine._commit_vent(imp, dest)
    assert ok and imp.location == dest
    # No other player recorded an arrival (secret relocation).
    for name, mem in engine.memory.items():
        if name != imp.name:
            assert not any(f"arrive in {dest}" in m for m in mem)


@pytest.mark.asyncio
async def test_emergency_meeting_fires_on_witnessed_kill():
    engine = GameEngine(GameConfig(seed=1, default_model="heuristic"))
    engine.setup()
    engine.round, engine.tick = 1, 1
    player = next(iter(engine.players.values()))
    player.emergencies_left = 1
    engine.memory[player.name] = ["R1.1: WITNESSED Blue kill Pink in Reactor!"]
    called = await engine._maybe_emergency(player)
    assert called and player.emergencies_left == 0


@pytest.mark.asyncio
async def test_reactor_fix_caps_at_required():
    engine = GameEngine(GameConfig(seed=1, default_model="heuristic", critical_fix_required=2))
    engine.setup()
    engine.round, engine.tick = 1, 1
    crew = next(p for p in engine.players.values() if p.role == Role.CREWMATE)
    crew.location = "Reactor"
    engine.critical = {"timer": 2, "fixes": 0, "required": 2}
    for _ in range(3):
        await engine._commit_fix(crew)
    assert engine.critical["fixes"] == 2  # extra fixes ignored


@pytest.mark.asyncio
async def test_reactor_suppresses_body_reports_until_fixed():
    engine = GameEngine(GameConfig(seed=1, default_model="heuristic"))
    engine.setup()
    engine.bodies = {"Pink": ("Reactor", "Blue")}
    finder = next(p for p in engine.players.values() if p.name not in ("Blue", "Pink"))
    finder.location, finder.alive = "Reactor", True
    engine.critical = {"timer": 2, "fixes": 0, "required": 2}
    assert engine._scan_for_report() is None
    engine.critical = None
    assert engine._scan_for_report() is not None


@pytest.mark.asyncio
async def test_comms_sabotage_blocks_reports():
    engine = GameEngine(GameConfig(seed=1, default_model="heuristic"))
    engine.setup()
    engine.bodies = {"Pink": ("Reactor", "Blue")}
    finder = next(p for p in engine.players.values() if p.name not in ("Blue", "Pink"))
    finder.location, finder.alive = "Reactor", True
    engine.sabotage = "comms"
    assert engine._scan_for_report() is None
    engine.sabotage = None
    assert engine._scan_for_report() is not None


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
