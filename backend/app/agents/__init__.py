"""Agent factory: build the right agent for a player's model spec."""
from __future__ import annotations

import random

from ..config import split_spec
from ..game.models import Role
from .base import Agent
from .heuristic_agent import HeuristicAgent
from .llm_agent import LLMAgent
from .providers import LLMError, client_for_spec


def make_agent(name: str, role: Role, spec: str, *, rng: random.Random | None = None) -> Agent:
    """Construct an agent from a model spec, falling back to heuristic on error.

    If a provider is unreachable (missing key, package, network), the player is
    quietly backed by the offline heuristic agent so the game still runs — the
    spec is preserved on the agent so the scoreboard still attributes results.
    """
    provider, _ = split_spec(spec)
    if provider == "heuristic":
        return HeuristicAgent(name, role, spec, rng=rng)
    try:
        client = client_for_spec(spec)
    except LLMError:
        client = None
    if client is None:
        return HeuristicAgent(name, role, spec, rng=rng)
    return LLMAgent(name, role, spec, client)


__all__ = ["Agent", "make_agent", "LLMAgent", "HeuristicAgent"]
