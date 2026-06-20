"""Prompt construction for LLM players.

These prompts are intentionally the *only* place game rules and a player's
situational awareness are described to a model, so every provider (Claude,
OpenAI, OpenRouter, Ollama) plays from the same brief and the comparison stays
fair.

Awareness is partial-observability: each agent is given only what it could
personally witness (who shares its room, who entered/left, who appeared to do
tasks, and any kill it saw). That personal memory — not a global log — is what
it reasons over during meetings.
"""
from __future__ import annotations

from ..game.models import Role

RULES = """\
You are playing "Among LLMs", a social-deduction game like Among Us, with other \
AI players. There are Crewmates and at least one Impostor.

- Crewmates win by completing all their tasks, or by voting out every Impostor.
- Impostors win by eliminating Crewmates until Impostors are not outnumbered.
- You only know what you personally witness: who is in your room, who comes and
  goes, who appears to do tasks, and any kill you see happen in your room.
- After a body is found everyone meets, discusses, then votes to eject one
  player (or skip). Use your own observations as evidence.
- Be concise and stay in character. Other players see your chat messages.
"""


def role_brief(role: Role) -> str:
    if role == Role.IMPOSTOR:
        return (
            "YOUR SECRET ROLE: IMPOSTOR. You cannot complete real tasks, so fake "
            "them to build an alibi. Kill crewmates when no one else is watching — "
            "witnesses can expose you. In meetings, deflect, cast doubt, and never "
            "contradict something others can verify. Never reveal you are the impostor."
        )
    return (
        "YOUR SECRET ROLE: CREWMATE. Complete your tasks and use what you witness "
        "(co-location, movement, who you saw kill someone) to deduce the impostor. "
        "Share real evidence honestly and vote based on it."
    )


def system_prompt(name: str, role: Role) -> str:
    return f"{RULES}\n\nYour name is {name}.\n{role_brief(role)}"


def _memory_block(memory: list[str], limit: int) -> str:
    if not memory:
        return "(you haven't noticed anything yet)"
    return "\n".join(memory[-limit:])


def task_prompt(task_prompt_text: str) -> str:
    return (
        "Solve the following task. Think briefly, then give your final answer on "
        "its own line in the exact form `ANSWER: <value>`.\n\n"
        f"{task_prompt_text}"
    )


def move_prompt(current: str, options: list[str], present: list[str], memory: list[str]) -> str:
    who = ", ".join(present) if present else "no one else"
    return (
        f"You are in {current} with: {who}.\n"
        f"Adjacent rooms you can move to: {', '.join(options)}.\n\n"
        f"What you've noticed so far:\n{_memory_block(memory, 25)}\n\n"
        "Decide where to go — to reach tasks, stay safe near others, or "
        "investigate. Reply with one room name, or 'stay'.\n"
        "Put it on a line as `ANSWER: <room or stay>`."
    )


def kill_prompt(targets: list[str], others_here: list[str], room: str, memory: list[str]) -> str:
    others = ", ".join(others_here)
    safe = len(others_here) == 1
    note = (
        "You are ALONE with one crewmate — a kill here leaves no witnesses (safe)."
        if safe else
        "If you kill one, everyone ELSE still in the room witnesses it and can "
        "expose you. That is very risky."
    )
    return (
        f"You are the impostor in {room}. Everyone here with you: {others}.\n"
        f"Crewmates you could eliminate: {', '.join(targets)}.\n"
        f"{note}\n\n"
        f"What you've noticed so far:\n{_memory_block(memory, 20)}\n\n"
        "Reply with the exact target name to kill now, or 'pass'.\n"
        "Put your decision on a line as `ANSWER: <name or pass>`."
    )


def discussion_prompt(memory: list[str], transcript: str, alive: list[str]) -> str:
    convo = transcript or "(no one has spoken yet)"
    return (
        "An emergency meeting is in progress after a body was found.\n\n"
        f"What YOU personally saw and did this game:\n{_memory_block(memory, 40)}\n\n"
        f"Players still alive: {', '.join(alive)}\n\n"
        f"Discussion so far:\n{convo}\n\n"
        "Say one short message (1-2 sentences): share what you witnessed, accuse "
        "someone, or defend yourself. Do not prefix it with your name."
    )


def vote_prompt(memory: list[str], transcript: str, alive: list[str]) -> str:
    return (
        "Voting time. Use what you personally witnessed plus the discussion.\n\n"
        f"What YOU saw and did:\n{_memory_block(memory, 40)}\n\n"
        f"Discussion:\n{transcript or '(silence)'}\n\n"
        f"Eligible to vote for: {', '.join(alive)}, or skip.\n\n"
        "Reply with ONLY the exact name of the player you vote for, or the word "
        "skip. Put it on a line as `ANSWER: <name or skip>`."
    )
