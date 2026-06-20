import React from "react";

const COLS = [
  ["model", "Model"],
  ["capability_score", "Score"],
  ["task_accuracy", "Task acc"],
  ["win_rate", "Win %"],
  ["vote_precision", "Vote prec"],
  ["deception_rate", "Deceive %"],
  ["games", "Games"],
];

const pct = (v) => `${Math.round(v * 100)}%`;

export default function Leaderboard({ rows, onReset }) {
  if (!rows || rows.length === 0) {
    return (
      <div className="panel">
        <h2>Leaderboard</h2>
        <p className="muted">No games recorded yet. Play or simulate some games.</p>
      </div>
    );
  }
  return (
    <div className="panel">
      <div className="row-between">
        <h2>Model leaderboard</h2>
        {onReset && (
          <button className="btn ghost" onClick={onReset}>
            Reset
          </button>
        )}
      </div>
      <table className="board">
        <thead>
          <tr>
            {COLS.map(([k, label]) => (
              <th key={k}>{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.model} className={i === 0 ? "leader" : ""}>
              <td className="mono">{r.model}</td>
              <td><strong>{r.capability_score}</strong></td>
              <td>{pct(r.task_accuracy)}</td>
              <td>{pct(r.win_rate)}</td>
              <td>{pct(r.vote_precision)}</td>
              <td>{pct(r.deception_rate)}</td>
              <td>{r.games}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small">
        Score = 45% task accuracy + 25% win rate + 20% vote precision + 10% survival.
      </p>
    </div>
  );
}
