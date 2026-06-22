import React, { useEffect, useRef, useState } from "react";
import GameSetup from "./components/GameSetup.jsx";
import GameView from "./components/GameView.jsx";
import EvalDashboard from "./components/EvalDashboard.jsx";
import CapabilityStrip from "./components/CapabilityStrip.jsx";
import {
  getProviders, getLeaderboard, resetLeaderboard, simulate, openGameSocket,
} from "./api.js";

const EMPTY_GAME = {
  map: {}, players: {}, chat: [], feed: [], thoughts: [],
  round: 0, phase: "lobby", winner: null, reason: "", reveal: null,
  sabotage: null, critical: null,
};

const DEFAULT_CONFIG = {
  num_players: 5,
  num_impostors: 1,
  max_rounds: 8,
  tasks_per_crewmate: 3,
  discussion_rounds: 2,
  event_delay: 0.5,
  default_model: "heuristic",
  players: [],
};

export default function App() {
  const [tab, setTab] = useState("play");
  const [providers, setProviders] = useState(null);
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [game, setGame] = useState(EMPTY_GAME);
  const [running, setRunning] = useState(false);
  const [board, setBoard] = useState([]);
  const [status, setStatus] = useState("");
  const wsRef = useRef(null);

  useEffect(() => {
    getProviders().then((p) => {
      setProviders(p);
      // Prefer Claude Opus as default if Anthropic is configured.
      if ((p.available || []).includes("claude")) {
        setConfig((c) => ({ ...c, default_model: p.default_model }));
      }
    });
    refreshBoard();
  }, []);

  const refreshBoard = () =>
    getLeaderboard().then((d) => setBoard(d.leaderboard || []));

  const applyEvent = (ev) => {
    setGame((g) => {
      const next = { ...g, round: ev.round ?? g.round, phase: ev.phase ?? g.phase };
      const d = ev.data || {};
      switch (ev.type) {
        case "game_start": {
          next.map = d.map || {};
          const players = {};
          (d.players || []).forEach((p) => {
            players[p.name] = {
              name: p.name, model: p.model, location: "Cafeteria",
              alive: true, tasks_completed: 0, tasks_total: 0,
            };
          });
          next.players = players;
          next.feed = [{ type: ev.type, message: ev.message }];
          next.chat = [];
          next.winner = null;
          next.reveal = null;
          return next;
        }
        case "phase_change": {
          next.sabotage = null; // lights/comms are fixed between rounds
          break;                // reactor (critical) persists until resolved
        }
        case "sabotage": {
          if (d.kind === "reactor") {
            next.critical = { fixes: 0, required: d.required, timer: d.timer };
          } else {
            next.sabotage = d.kind;
          }
          break;
        }
        case "fix": {
          if (next.critical) next.critical = { ...next.critical, fixes: d.fixes };
          break;
        }
        case "info": {
          if (d.reactor === "stabilized") next.critical = null;
          break;
        }
        case "move":
        case "vent": {
          const p = next.players[d.player];
          if (p) next.players = { ...next.players, [d.player]: { ...p, location: d.to } };
          break;
        }
        case "thought": {
          next.thoughts = [
            ...g.thoughts,
            { actor: d.actor, action: d.action, text: d.text },
          ].slice(-80);
          break;
        }
        case "task_result": {
          const p = next.players[d.player];
          if (p)
            next.players = {
              ...next.players,
              [d.player]: {
                ...p,
                tasks_completed: d.tasks_completed,
                tasks_total: d.tasks_total,
              },
            };
          break;
        }
        case "kill": {
          const p = next.players[d.victim];
          if (p) next.players = { ...next.players, [d.victim]: { ...p, alive: false } };
          break;
        }
        case "ejection": {
          if (d.player) {
            const p = next.players[d.player];
            if (p) next.players = { ...next.players, [d.player]: { ...p, alive: false } };
          }
          break;
        }
        case "chat": {
          next.chat = [...g.chat, { speaker: d.speaker, text: d.text }];
          break;
        }
        case "game_end": {
          next.winner = d.winner;
          next.reason = d.reason;
          next.reveal = d.reveal;
          break;
        }
        default:
          break;
      }
      // Append to feed (skip moves + thoughts; those have their own surfaces).
      if (ev.type !== "move" && ev.type !== "thought") {
        next.feed = [...g.feed, { type: ev.type, message: ev.message }].slice(-200);
      }
      return next;
    });
  };

  const startGame = () => {
    setGame(EMPTY_GAME);
    setRunning(true);
    setStatus("Connecting…");
    const ws = openGameSocket(config);
    wsRef.current = ws;
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === "eval" || msg.type === "leaderboard") {
        setBoard(msg.data || []); // live eval snapshot during the game
        return;
      }
      if (msg.type === "error") {
        setStatus("Error: " + msg.message);
        return;
      }
      applyEvent(msg);
      if (msg.type === "game_end") setStatus("Game finished.");
    };
    ws.onclose = () => {
      setRunning(false);
      refreshBoard();
    };
    ws.onerror = () => setStatus("Connection error — is the backend running?");
  };

  const runSimulation = async (games) => {
    setRunning(true);
    setStatus(`Simulating ${games} games…`);
    try {
      const res = await simulate({ ...config, games });
      setBoard(res.leaderboard || []);
      setStatus(`Done. ${games} games simulated.`);
      setTab("leaderboard");
    } catch (e) {
      setStatus("Simulation failed: " + e.message);
    } finally {
      setRunning(false);
    }
  };

  const doReset = async () => {
    await resetLeaderboard();
    refreshBoard();
  };

  return (
    <div className="app">
      <header className="topbar">
        <h1>🔭 Among LLMs</h1>
        <nav>
          <button className={tab === "play" ? "active" : ""} onClick={() => setTab("play")}>
            Play
          </button>
          <button
            className={tab === "leaderboard" ? "active" : ""}
            onClick={() => { setTab("leaderboard"); refreshBoard(); }}
          >
            Eval dashboard
          </button>
        </nav>
        <span className="tagline">social-deduction eval for agentic LLMs</span>
      </header>

      {status && <div className="status">{status}</div>}

      {tab === "play" && (
        <div className="play">
          <GameSetup
            providers={providers}
            config={config}
            setConfig={setConfig}
            onStart={startGame}
            onSimulate={runSimulation}
            running={running}
          />
          {(game.round > 0 || running) && (
            <>
              <CapabilityStrip rows={board} />
              <GameView game={game} />
            </>
          )}
        </div>
      )}

      {tab === "leaderboard" && (
        <EvalDashboard rows={board} onReset={doReset} live={running} />
      )}

      <footer className="foot">
        Mix Claude · OpenAI · OpenRouter · Ollama players. Capability tasks +
        social deduction → per-model scores.
      </footer>
    </div>
  );
}
