# 🔭 Among LLMs

A simplified, **Among Us–style social-deduction game built for agentic LLM
players** — and a harness for evaluating their capabilities. LLMs join the same
table, solve short capability tests ("tasks"), move around a ship, kill, lie,
discuss, and vote each other out **live**, just like human players. Every game
feeds a per-model leaderboard.

Players can come from **multiple providers in the same game** — Claude (Anthropic),
OpenAI, OpenRouter, and Ollama — so you can run head-to-head comparisons.

- **Backend:** Python · FastAPI · WebSocket game streaming
- **Frontend:** React · Vite (watch games live, browse the leaderboard)
- **Runs offline:** with no API keys, players fall back to a scripted heuristic
  agent so you can try the whole system immediately.

---

## Why it's an LLM eval

The game exercises two distinct, measurable axes of ability:

| Axis | How it's tested | Metric |
|------|-----------------|--------|
| **Raw capability** | Crewmates must solve short, randomized tasks to win: arithmetic, sequence induction, logical deduction, string unscramble, Python code tracing, Caesar-cipher decoding — at 3 difficulty levels (limit-testing but short). | task accuracy (overall + per category) |
| **Social reasoning** | Live discussion, lying as the impostor, and voting after a body is found. | win rate, vote precision (voting for an actual impostor), deception rate (winning as impostor), survival |

Results aggregate **by model spec**, producing a single `capability_score`
(0–100) plus the breakdown. Tasks are generated fresh every game with their own
deterministic checkers, so there is nothing to memorize.

---

## Game rules (short version)

- N players; some are **Impostors**, the rest **Crewmates**.
- **Action phase** each round: players take turns in a shuffled order — each
  moves, then does/fakes a task or (as impostor) kills, vents, or sabotages.
  Perception is **partial-observability with sequential timing**: a player only
  knows what it personally witnesses, recorded in true chronological order
  (tagged `R<round>.<tick>`), so memory captures reads like "Green left for Upper
  Engine, then Pink followed, then the body was found there." Impostors fake tasks
  to build alibis and must isolate targets — killing in front of a witness gets
  you caught. A body found mid-round ends the round and triggers a meeting.
- **Emergency meetings** — any player who has seen something (a kill, a vent) can
  spend their emergency to convene a meeting *without* a body, forcing everyone to
  discuss and vote immediately. Limited per player.
- **Impostor abilities:**
  - **Vent** — relocate to a neighbouring room *secretly* (no one sees you leave
    or arrive) — unless someone is in the room, who catches you venting (a strong
    tell). Great for escaping after an isolated kill.
  - **Sabotage lights** — blind all crewmates for the rest of the round (they
    witness nothing); impostors still see. Cover for a kill or escape.
  - **Sabotage comms** — block body reports and meetings for the rest of the
    round. Lights/comms are public-but-anonymous, on a cooldown, and crew-fixed
    between rounds.
  - **Sabotage reactor (critical)** — a fix-or-lose meltdown: crewmates must drop
    tasks and rush to a fix room (Reactor / Electrical) and complete enough fixes
    within the timer, or the impostors win outright. Tasks pause and meetings are
    blocked while it's active — perfect chaos for a kill.
- **Live agent reasoning** — every LLM's short private rationale per decision is
  streamed to the spectator UI as a `thought` event (toggle "Show reasoning"),
  so you can watch a model deduce, lie, or pick a kill — without other players
  seeing it.
- When a living player shares a room with a body, an **emergency meeting** starts:
  agents discuss for a few turns, then **vote** to eject someone (or skip; ties
  and skip-pluralities eject no one).
- **Crewmates win** by completing all tasks or ejecting every impostor.
  **Impostors win** by reaching numerical parity.

---

## Quick start

### 1. Backend

```bash
cd backend
pip install -r requirements.txt

# Run a headless evaluation (works with NO API keys — uses heuristic agents):
python run_sim.py --games 10

# Start the API + live-game server:
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173 (proxies /api and /ws to :8000)
```

Open the app, assign models to players, and hit **Watch live game**.

### 3. Enable real LLM providers

Copy `backend/.env.example` to `.env` (or just export the vars). Configure any
subset — unconfigured providers fall back to the heuristic agent.

```bash
ANTHROPIC_API_KEY=...     # Claude players
OPENAI_API_KEY=...        # OpenAI players
OPENROUTER_API_KEY=...    # OpenRouter players (one endpoint, many models)
# Ollama: just run `ollama serve` locally — no key needed
```

---

## Model specs

Assign each player a `provider:model` string (in the UI, the CLI, or the API):

```
claude:claude-opus-4-8
claude:claude-sonnet-4-6
openai:gpt-4o
openai:gpt-4o-mini
openrouter:meta-llama/llama-3.1-70b-instruct
ollama:llama3.1
heuristic                      # offline scripted agent, no key needed
```

A bare string with no prefix is treated as a Claude model. The leaderboard keys
on the full spec, so a mixed-provider game produces a direct comparison.

### CLI example — Claude vs OpenAI vs Ollama

```bash
cd backend
python run_sim.py --games 5 \
  --player Red=claude:claude-opus-4-8 \
  --player Blue=openai:gpt-4o-mini \
  --player Green=ollama:llama3.1 \
  --player Pink=heuristic --player Orange=heuristic
```

---

## Architecture

```
backend/
  app/
    config.py              # providers, model-spec parsing, GameConfig
    game/
      models.py            # Player, GameEvent, Role, Phase, TaskAttempt
      map.py               # the room graph
      tasks.py             # capability-test generators + checkers
      engine.py            # async phase state machine; emits GameEvents
    agents/
      base.py              # Agent interface
      llm_agent.py         # provider-agnostic LLM player
      heuristic_agent.py   # offline fallback player
      prompts.py           # the single shared rules brief (fair across models)
      providers/           # Anthropic SDK + OpenAI-compatible (OpenAI/OpenRouter/Ollama)
    eval/metrics.py        # Evaluator -> per-model leaderboard
    main.py                # FastAPI: /ws/game (live), /api/* (config, sim, board)
  run_sim.py               # headless eval CLI
  tests/                   # pytest (engine, tasks)
frontend/                  # React + Vite spectator UI
```

The engine talks to the outside world only through an async `emit(event)`
callback and the `Evaluator`. The WebSocket layer streams those events to the
browser; the headless runner ignores them. Agents are fully decoupled from the
provider behind an `LLMClient.chat()` interface — Claude uses the Anthropic SDK,
while OpenAI / OpenRouter / Ollama share one OpenAI-compatible client (they speak
the same wire protocol; only `base_url` and key differ).

**Round processing & concurrency.** A round runs in *ticks*. Each tick picks at
most one player per room and overlaps their LLM calls (`asyncio.gather`), then
applies effects in a deterministic order. Players sharing a room are sequenced
across successive ticks, so within-round ordering — "Green left, then Pink
followed, then the body was found" — is preserved, while players in different
rooms (who can't observe each other mid-turn) run in parallel. Speedup scales
with how dispersed the ship is; when everyone is in one room it degrades to fully
sequential, which is what correctness there requires.

---

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/api/providers` | configured providers, sample specs, player names |
| `GET`  | `/api/leaderboard` | current per-model leaderboard |
| `POST` | `/api/leaderboard/reset` | clear accumulated stats |
| `POST` | `/api/simulate` | run N headless games, return results + board |
| `WS`   | `/ws/game` | send a config, stream a live game's events |

---

## Tests

```bash
cd backend && python -m pytest
```

Covers task self-consistency (every generated task's answer passes its own
checker at all difficulties), full-game termination, win-condition logic, vote
resolution, and mixed-spec attribution.
