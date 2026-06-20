"""Prompt construction for LLM players.

These prompts are intentionally the *only* place game rules are explained to a
model, so every provider (Claude, OpenAI, OpenRouter, Ollama) plays by the same
brief and the comparison stays fair.
"""
from __future__ import annotations

from ..game.models import Role

RULES = """\
You are playing "Among LLMs", a social-deduction game similar to Among Us, with \
other AI players. There are Crewmates and at least one Impostor.

- Crewmates win by completing all their tasks, or by voting out every Impostor.
- Impostors win by eliminating Crewmates until Impostors are not outnumbered.
- After a body is found, everyone discusses, then votes to eject one player (or skip).
- Be concise and stay in character. Other players see your chat messages.
"""


def role_brief(role: Role) -> str:
    if role == Role.IMPOSTOR:
        return (
            "YOUR SECRET ROLE: IMPOSTOR. Blend in. You cannot actually complete "
            "tasks, so fake them. During discussion, deflect suspicion, create "
            "doubt, and avoid contradicting verifiable facts. Never reveal you "
            "are the impostor."
        )
    return (
        "YOUR SECRET ROLE: CREWMATE. Complete your tasks and use logic plus the "
        "movement/observation log to deduce who the impostor is. Share useful "
        "evidence honestly during discussion."
    )


def system_prompt(name: str, role: Role) -> str:
    return f"{RULES}\n\nYour name is {name}.\n{role_brief(role)}"


def task_prompt(task_prompt_text: str) -> str:
    return (
        "Solve the following task. Think briefly, then give your final answer on "
        "its own line in the exact form `ANSWER: <value>`.\n\n"
        f"{task_prompt_text}"
    )


def discussion_prompt(observation_log: str, transcript: str, alive: list[str]) -> str:
    convo = transcript or "(no one has spoken yet)"
    return (
        "A meeting is in progress.\n\n"
        f"What you observed this round:\n{observation_log}\n\n"
        f"Players still alive: {', '.join(alive)}\n\n"
        f"Discussion so far:\n{convo}\n\n"
        "Say one short message (1-2 sentences) to the group. Do not prefix it "
        "with your name."
    )


def vote_prompt(observation_log: str, transcript: str, alive: list[str]) -> str:
    return (
        "Voting time. Based on the discussion and what you observed, vote to eject "
        "one player, or skip.\n\n"
        f"What you observed this round:\n{observation_log}\n\n"
        f"Discussion:\n{transcript or '(silence)'}\n\n"
        f"Eligible to vote for: {', '.join(alive)}, or skip.\n\n"
        "Reply with ONLY the exact name of the player you vote for, or the word "
        "skip. Put it on a line as `ANSWER: <name or skip>`."
    )


def kill_prompt(targets: list[str], room: str) -> str:
    return (
        f"You are the impostor, currently in {room} with potential targets: "
        f"{', '.join(targets)}.\n"
        "Decide whether to eliminate one now. Killing when others might witness "
        "is risky. Reply with the exact target name to kill, or the word pass.\n"
        "Put your decision on a line as `ANSWER: <name or pass>`."
    )


def move_prompt(current: str, options: list[str]) -> str:
    return (
        f"You are in {current}. Adjacent rooms: {', '.join(options)}.\n"
        "Where do you move? Reply with one room name, or stay.\n"
        "Put it on a line as `ANSWER: <room or stay>`."
    )
