import React from "react";

const pct = (v) => `${Math.round((v || 0) * 100)}%`;

// Compact, live capability strip — docked under the game board so you can watch
// the match and the scoreboard side by side. Reads the same live eval state.
export default function CapabilityStrip({ rows }) {
  if (!rows || rows.length === 0) return null;
  return (
    <div className="cap-strip">
      <span className="cap-title">live eval <span className="live-dot">●</span></span>
      <div className="cap-chips">
        {rows.map((r, i) => (
          <div className={`cap-chip ${i === 0 ? "lead" : ""}`} key={r.model}>
            <span className="cap-model mono" title={r.model}>{r.model}</span>
            <span className="cap-score">{r.capability_score}</span>
            <span className="cap-sub">
              tasks {pct(r.task_accuracy)} · win {pct(r.win_rate)} · {r.games}g
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
