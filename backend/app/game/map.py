"""A small room graph players move through.

Kept deliberately tiny so co-location (the precondition for kills and the basis
for alibis) happens often enough to drive the social-deduction game.
"""
from __future__ import annotations

ROOMS: dict[str, list[str]] = {
    "Cafeteria": ["Upper Engine", "Storage", "MedBay"],
    "Upper Engine": ["Cafeteria", "Reactor"],
    "Reactor": ["Upper Engine", "Storage"],
    "Storage": ["Cafeteria", "Reactor", "Electrical"],
    "Electrical": ["Storage", "MedBay"],
    "MedBay": ["Cafeteria", "Electrical"],
}

ALL_ROOMS = list(ROOMS)
START_ROOM = "Cafeteria"
# Rooms where a reactor (critical) sabotage is fixed.
FIX_ROOMS = ["Reactor", "Electrical"]


def neighbors(room: str) -> list[str]:
    return ROOMS.get(room, [])


def step_toward(src: str, targets: list[str]) -> str | None:
    """First room to move to on a shortest path from ``src`` to any target.

    Returns None if already at a target or no path exists. Used to guide players
    toward reactor fix rooms during a critical sabotage.
    """
    from collections import deque

    if src in targets:
        return None
    parent: dict[str, str | None] = {src: None}
    q = deque([src])
    while q:
        cur = q.popleft()
        for nb in ROOMS.get(cur, []):
            if nb in parent:
                continue
            parent[nb] = cur
            if nb in targets:
                step = nb
                while parent[step] != src:
                    step = parent[step]  # walk back to the first hop
                return step
            q.append(nb)
    return None
