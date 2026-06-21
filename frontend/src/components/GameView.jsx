import React, { useEffect, useRef, useState } from "react";
import { COLORS } from "./GameSetup.jsx";

function Dot({ name, dead }) {
  return (
    <span
      className={`crew ${dead ? "dead" : ""}`}
      title={name}
      style={{ background: COLORS[name] || "#888" }}
    >
      {dead ? "✝" : ""}
    </span>
  );
}

function MapView({ map, players }) {
  const rooms = Object.keys(map || {});
  return (
    <div className="map">
      {rooms.map((room) => {
        const here = Object.values(players).filter((p) => p.location === room);
        return (
          <div className="room" key={room}>
            <div className="room-name">{room}</div>
            <div className="room-crew">
              {here.map((p) => (
                <Dot key={p.name} name={p.name} dead={!p.alive} />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function PlayerList({ players, reveal }) {
  const roleByName = {};
  (reveal || []).forEach((r) => (roleByName[r.name] = r.role));
  return (
    <div className="player-list">
      {Object.values(players).map((p) => {
        const pct = p.tasks_total ? (p.tasks_completed / p.tasks_total) * 100 : 0;
        const role = roleByName[p.name];
        return (
          <div className={`pcard ${p.alive ? "" : "is-dead"}`} key={p.name}>
            <div className="pcard-head">
              <span className="dot" style={{ background: COLORS[p.name] }} />
              <span className="pname">{p.name}</span>
              {role && (
                <span className={`role ${role}`}>{role}</span>
              )}
              {!p.alive && !role && <span className="role dead-tag">out</span>}
            </div>
            <div className="mono tiny">{p.model}</div>
            {p.tasks_total > 0 && (
              <div className="bar">
                <div className="bar-fill" style={{ width: `${pct}%` }} />
                <span className="bar-label">
                  {p.tasks_completed}/{p.tasks_total} tasks
                </span>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function ChatPanel({ chat }) {
  const end = useRef(null);
  useEffect(() => end.current?.scrollIntoView({ behavior: "smooth" }), [chat]);
  return (
    <div className="chat">
      <h3>Discussion</h3>
      <div className="chat-body">
        {chat.length === 0 && <p className="muted small">No meeting yet.</p>}
        {chat.map((m, i) => (
          <div className="msg" key={i}>
            <span className="dot small" style={{ background: COLORS[m.speaker] }} />
            <span className="mspk">{m.speaker}</span>
            <span className="mtext">{m.text}</span>
          </div>
        ))}
        <div ref={end} />
      </div>
    </div>
  );
}

function ThoughtsPanel({ thoughts }) {
  const end = useRef(null);
  useEffect(() => end.current?.scrollIntoView({ behavior: "smooth" }), [thoughts]);
  return (
    <div className="thoughts">
      <h3>🧠 Agent reasoning <span className="muted small">(private — spoilers)</span></h3>
      <div className="thoughts-body">
        {thoughts.length === 0 && <p className="muted small">No thoughts yet.</p>}
        {thoughts.map((t, i) => (
          <div className="thought" key={i}>
            <span className="dot small" style={{ background: COLORS[t.actor] }} />
            <span className="tspk">{t.actor}</span>
            <span className="taction">{t.action}</span>
            <span className="ttext">{t.text}</span>
          </div>
        ))}
        <div ref={end} />
      </div>
    </div>
  );
}

const ICON = {
  task_result: "🧩", kill: "🔪", body_reported: "🚨", vote: "🗳️",
  ejection: "🚪", move: "🚶", chat: "💬", game_end: "🏁", meeting_start: "📣",
  phase_change: "⏱️", game_start: "🚀", vent: "🟪", sabotage: "⚠️",
};

function EventFeed({ feed }) {
  const end = useRef(null);
  useEffect(() => end.current?.scrollIntoView({ behavior: "smooth" }), [feed]);
  return (
    <div className="feed">
      <h3>Event log</h3>
      <div className="feed-body">
        {feed.map((e, i) => (
          <div className={`fline ${e.type}`} key={i}>
            <span className="ficon">{ICON[e.type] || "•"}</span>
            <span className="fmsg">{e.message}</span>
          </div>
        ))}
        <div ref={end} />
      </div>
    </div>
  );
}

export default function GameView({ game }) {
  const { map, players, chat, feed, thoughts, round, phase, winner, reason, sabotage } = game;
  const [showThoughts, setShowThoughts] = useState(true);
  return (
    <div className="gameview">
      <div className="status-bar">
        <span className="pill">Round {round}</span>
        <span className={`pill phase-${phase}`}>{phase}</span>
        {sabotage && <span className="pill sabotage">⚠️ {sabotage} sabotaged</span>}
        {winner && (
          <span className={`pill win ${winner}`}>
            {winner === "impostors" ? "🔴 Impostors win" : "🔵 Crewmates win"} — {reason}
          </span>
        )}
        <button className="btn ghost tiny-btn" onClick={() => setShowThoughts((s) => !s)}>
          {showThoughts ? "Hide reasoning" : "Show reasoning"}
        </button>
      </div>
      <div className="cols">
        <div className="col-left">
          <MapView map={map} players={players} />
          <PlayerList players={players} reveal={game.reveal} />
        </div>
        <div className="col-right">
          {showThoughts && <ThoughtsPanel thoughts={thoughts} />}
          <ChatPanel chat={chat} />
          <EventFeed feed={feed} />
        </div>
      </div>
    </div>
  );
}
