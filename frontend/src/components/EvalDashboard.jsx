import React from "react";

const pct = (v) => `${Math.round((v || 0) * 100)}%`;

// Ordered metric columns for the main table.
const COLS = [
  ["capability_score", "Score", (r) => r.capability_score],
  ["task_accuracy", "Task acc", (r) => pct(r.task_accuracy)],
  ["win_rate", "Win%", (r) => pct(r.win_rate)],
  ["crew_win_rate", "Crew%", (r) => pct(r.crew_win_rate)],
  ["impostor_win_rate", "Impostor%", (r) => pct(r.impostor_win_rate)],
  ["vote_precision", "Vote prec", (r) => pct(r.vote_precision)],
  ["survival_rate", "Survival", (r) => pct(r.survival_rate)],
  ["games", "Games", (r) => r.games],
];

const CATS = ["arithmetic", "sequence", "deduction", "unscramble", "code_trace", "cipher"];

function Bar({ value }) {
  const v = Math.round((value || 0) * 100);
  return (
    <div className="ebar">
      <div className="ebar-fill" style={{ width: `${v}%` }} />
      <span className="ebar-label">{v}%</span>
    </div>
  );
}

function SummaryCards({ rows }) {
  const attempts = rows.reduce((a, r) => a + (r.task_attempts || 0), 0);
  const games = rows.reduce((m, r) => Math.max(m, r.games || 0), 0);
  const leader = rows[0];
  return (
    <div className="cards">
      <div className="card"><div className="card-n">{rows.length}</div><div className="card-l">models</div></div>
      <div className="card"><div className="card-n">{games}</div><div className="card-l">games</div></div>
      <div className="card"><div className="card-n">{attempts}</div><div className="card-l">task attempts</div></div>
      <div className="card">
        <div className="card-n mono small">{leader ? leader.model : "—"}</div>
        <div className="card-l">top model{leader ? ` · ${leader.capability_score}` : ""}</div>
      </div>
    </div>
  );
}

export default function EvalDashboard({ rows, onReset, live }) {
  if (!rows || rows.length === 0) {
    return (
      <div className="panel">
        <h2>Eval dashboard</h2>
        <p className="muted">No games recorded yet. Watch or simulate games — this
          updates live as tasks are attempted and votes are cast.</p>
      </div>
    );
  }
  return (
    <div className="panel">
      <div className="row-between">
        <h2>
          Eval dashboard {live && <span className="live-dot" title="updating live">● live</span>}
        </h2>
        {onReset && <button className="btn ghost" onClick={onReset}>Reset</button>}
      </div>

      <SummaryCards rows={rows} />

      <table className="board">
        <thead>
          <tr>
            <th>Model</th>
            {COLS.map(([k, label]) => <th key={k}>{label}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.model} className={i === 0 ? "leader" : ""}>
              <td className="mono">{r.model}</td>
              {COLS.map(([k, , get]) => (
                <td key={k}>{k === "capability_score" ? <strong>{get(r)}</strong> : get(r)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>

      <h3>Per-capability task accuracy</h3>
      <div className="cat-grid">
        {rows.map((r) => (
          <div className="cat-card" key={r.model}>
            <div className="mono small cat-model">{r.model}</div>
            {CATS.map((c) => (
              <div className="cat-row" key={c}>
                <span className="cat-name">{c}</span>
                <Bar value={(r.category_accuracy || {})[c]} />
              </div>
            ))}
          </div>
        ))}
      </div>

      <p className="muted small">
        Score = 45% task accuracy + 25% win rate + 20% vote precision + 10% survival.
        Impostor% is deception (win rate when cast as impostor); Vote prec is how
        often a crewmate voted for a real impostor.
      </p>
    </div>
  );
}
