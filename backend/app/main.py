"""FastAPI app: live game streaming over WebSocket + REST for config/leaderboard."""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import DEFAULT_MODEL, GameConfig, PROVIDERS, available_providers
from .eval.metrics import Evaluator
from .game.engine import PLAYER_NAMES, GameEngine

app = FastAPI(title="Among LLMs", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One process-wide evaluator so the leaderboard accumulates across games.
EVALUATOR = Evaluator()

SAMPLE_SPECS = [
    "claude:claude-opus-4-8",
    "claude:claude-sonnet-4-6",
    "openai:gpt-4o",
    "openai:gpt-4o-mini",
    "openrouter:meta-llama/llama-3.1-70b-instruct",
    "ollama:llama3.1",
    "heuristic",
]


def _config_from_payload(payload: dict) -> GameConfig:
    players = payload.get("players") or []
    assignments = {
        p["name"]: p["model"]
        for p in players
        if p.get("name") and p.get("model")
    }
    num_players = payload.get("num_players") or (len(assignments) or 5)
    return GameConfig(
        num_players=min(max(int(num_players), 3), len(PLAYER_NAMES)),
        num_impostors=max(1, int(payload.get("num_impostors", 1))),
        max_rounds=int(payload.get("max_rounds", 8)),
        tasks_per_crewmate=int(payload.get("tasks_per_crewmate", 3)),
        discussion_rounds=int(payload.get("discussion_rounds", 2)),
        event_delay=float(payload.get("event_delay", 0.0)),
        seed=payload.get("seed"),
        model_assignments=assignments,
        default_model=payload.get("default_model", DEFAULT_MODEL),
    )


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/api/providers")
async def providers() -> dict:
    """Tell the UI which providers are configured and how to spec a player."""
    return {
        "default_model": DEFAULT_MODEL,
        "available": available_providers(),
        "all": [p for p in PROVIDERS if p != "anthropic"],
        "sample_specs": SAMPLE_SPECS,
        "player_names": PLAYER_NAMES,
    }


@app.get("/api/leaderboard")
async def leaderboard() -> dict:
    return {"leaderboard": EVALUATOR.leaderboard()}


@app.post("/api/leaderboard/reset")
async def reset_leaderboard() -> dict:
    EVALUATOR.stats.clear()
    return {"status": "cleared"}


@app.post("/api/simulate")
async def simulate(payload: dict) -> JSONResponse:
    """Run one or more headless games (no streaming) and return summaries."""
    games = min(int(payload.get("games", 1)), 50)
    results = []
    for _ in range(games):
        cfg = _config_from_payload(payload)
        engine = GameEngine(cfg, evaluator=EVALUATOR)
        result = await engine.run()
        results.append(
            {"winner": result.winner, "reason": result.reason, "rounds": result.rounds,
             "players": result.players}
        )
    return JSONResponse(
        {"results": results, "leaderboard": EVALUATOR.leaderboard()}
    )


@app.websocket("/ws/game")
async def ws_game(ws: WebSocket) -> None:
    """Stream a single live game. First client message is the game config."""
    await ws.accept()
    try:
        payload = await ws.receive_json()
    except (WebSocketDisconnect, Exception):
        await ws.close()
        return

    cfg = _config_from_payload(payload)
    # Give spectators a watchable pace if the client didn't set one.
    if cfg.event_delay == 0.0:
        cfg.event_delay = 0.4

    async def emit(event) -> None:
        await ws.send_json(event.to_dict())

    engine = GameEngine(cfg, emit=emit, evaluator=EVALUATOR)
    try:
        await engine.run()
        await ws.send_json({"type": "leaderboard", "data": EVALUATOR.leaderboard()})
    except WebSocketDisconnect:
        return
    except Exception as exc:  # surface engine errors to the client instead of dropping
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# Serve the built frontend if present (single-container deploys).
_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="frontend")
