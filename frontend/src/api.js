// Thin API helpers. In dev, Vite proxies /api and /ws to the backend.

export async function getProviders() {
  const r = await fetch("/api/providers");
  return r.json();
}

export async function getLeaderboard() {
  const r = await fetch("/api/leaderboard");
  return r.json();
}

export async function resetLeaderboard() {
  await fetch("/api/leaderboard/reset", { method: "POST" });
}

export async function simulate(payload) {
  const r = await fetch("/api/simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return r.json();
}

// Open a live game WebSocket. Returns the socket; caller wires onmessage.
export function openGameSocket(config) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/game`);
  ws.addEventListener("open", () => ws.send(JSON.stringify(config)));
  return ws;
}
