import React from "react";

const COLORS = {
  Red: "#c51111", Blue: "#132ed1", Green: "#117f2d", Pink: "#ed54ba",
  Orange: "#ef7d0d", Yellow: "#f5f557", Cyan: "#39fed4", Lime: "#50ef39",
  White: "#d6e0f0", Black: "#3f474e",
};

export default function GameSetup({ providers, config, setConfig, onStart, onSimulate, running }) {
  const names = providers?.player_names || [];
  const samples = providers?.sample_specs || ["heuristic"];

  const update = (k, v) => setConfig({ ...config, [k]: v });

  const setPlayer = (name, model) => {
    const players = config.players.filter((p) => p.name !== name);
    players.push({ name, model });
    update("players", players);
  };

  const playerModel = (name) =>
    config.players.find((p) => p.name === name)?.model || config.default_model;

  const activeNames = names.slice(0, config.num_players);

  return (
    <div className="panel">
      <h2>Set up a game</h2>

      <div className="grid2">
        <label>
          Players
          <input
            type="number" min="3" max={names.length} value={config.num_players}
            onChange={(e) => update("num_players", Number(e.target.value))}
          />
        </label>
        <label>
          Impostors
          <input
            type="number" min="1" max={Math.max(1, config.num_players - 2)}
            value={config.num_impostors}
            onChange={(e) => update("num_impostors", Number(e.target.value))}
          />
        </label>
        <label>
          Max rounds
          <input type="number" min="1" max="20" value={config.max_rounds}
            onChange={(e) => update("max_rounds", Number(e.target.value))} />
        </label>
        <label>
          Tasks / crewmate
          <input type="number" min="1" max="6" value={config.tasks_per_crewmate}
            onChange={(e) => update("tasks_per_crewmate", Number(e.target.value))} />
        </label>
        <label>
          Discussion turns
          <input type="number" min="1" max="4" value={config.discussion_rounds}
            onChange={(e) => update("discussion_rounds", Number(e.target.value))} />
        </label>
        <label>
          Default model
          <input list="specs" value={config.default_model}
            onChange={(e) => update("default_model", e.target.value)} />
        </label>
      </div>

      <datalist id="specs">
        {samples.map((s) => <option key={s} value={s} />)}
      </datalist>

      <h3>Assign models to players</h3>
      <p className="muted small">
        Format <code>provider:model</code> — e.g. <code>claude:claude-opus-4-8</code>,{" "}
        <code>openai:gpt-4o</code>, <code>openrouter:…</code>, <code>ollama:llama3.1</code>,
        or <code>heuristic</code> (offline). Providers configured:{" "}
        <strong>{(providers?.available || []).join(", ") || "heuristic only"}</strong>.
      </p>

      <div className="players">
        {activeNames.map((name) => (
          <div className="player-row" key={name}>
            <span className="dot" style={{ background: COLORS[name] }} />
            <span className="pname">{name}</span>
            <input
              list="specs"
              value={playerModel(name)}
              onChange={(e) => setPlayer(name, e.target.value)}
              placeholder={config.default_model}
            />
          </div>
        ))}
      </div>

      <div className="actions">
        <button className="btn primary" onClick={onStart} disabled={running}>
          {running ? "Game running…" : "▶ Watch live game"}
        </button>
        <button className="btn" onClick={() => onSimulate(10)} disabled={running}>
          Simulate 10 (headless)
        </button>
      </div>
    </div>
  );
}

export { COLORS };
