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


def neighbors(room: str) -> list[str]:
    return ROOMS.get(room, [])
