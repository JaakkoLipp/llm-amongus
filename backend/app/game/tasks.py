"""Capability-test tasks — the "tasks" crewmates complete.

Each task is a short, self-contained probe of a distinct LLM capability:
arithmetic, sequence induction, logical deduction, string manipulation, code
tracing, and constraint satisfaction. Tasks are *generated* with randomized
parameters every game so players cannot memorize answers, and each ships its own
deterministic checker. Completing a task = answering its probe correctly, which
is exactly the per-capability signal the evaluation harness records.

Difficulty (1-3) scales the size/length of the probe, letting you limit-test how
far a model holds up before it breaks.
"""
from __future__ import annotations

import random
import re
import string
from dataclasses import dataclass
from typing import Callable


@dataclass
class Task:
    category: str
    difficulty: int
    prompt: str
    expected: str
    check: Callable[[str], bool]


def _norm(s: str) -> str:
    return s.strip().lower().strip(".!? ")


def _exact(expected: str) -> Callable[[str], bool]:
    target = _norm(expected)

    def check(answer: str) -> bool:
        a = _norm(extract_answer(answer))
        return a == target

    return check


def extract_answer(raw: str) -> str:
    """Pull the final answer out of a model's free-form reply.

    Players are told to end with ``ANSWER: <x>``; we honour that, then fall back
    to the last non-empty line so a terse model still scores.
    """
    if not raw:
        return ""
    m = re.findall(r"ANSWER\s*[:\-]\s*(.+)", raw, flags=re.IGNORECASE)
    if m:
        return m[-1].strip()
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return lines[-1] if lines else raw.strip()


# --- individual task generators ------------------------------------------------

def _arithmetic(rng: random.Random, diff: int) -> Task:
    n = 2 + diff  # number of terms
    nums = [rng.randint(2, 9 * diff + 9) for _ in range(n)]
    ops = [rng.choice(["+", "-", "*"]) for _ in range(n - 1)]
    expr = str(nums[0])
    for op, val in zip(ops, nums[1:]):
        expr += f" {op} {val}"
    result = eval(expr)  # noqa: S307 - expr built only from our own ints/ops
    prompt = f"Evaluate this expression exactly: {expr}"
    return Task("arithmetic", diff, prompt, str(result), _exact(str(result)))


def _sequence(rng: random.Random, diff: int) -> Task:
    start = rng.randint(1, 9)
    step = rng.randint(2, 4 + diff)
    kind = rng.choice(["arith", "geom", "square"])
    if kind == "arith":
        seq = [start + step * i for i in range(4 + diff)]
        nxt = seq[-1] + step
    elif kind == "geom":
        ratio = rng.randint(2, 3)
        seq = [start * (ratio ** i) for i in range(4)]
        nxt = seq[-1] * ratio
    else:
        base = rng.randint(1, 3 + diff)
        seq = [(base + i) ** 2 for i in range(4 + diff)]
        nxt = (base + len(seq)) ** 2
    shown = ", ".join(str(x) for x in seq)
    prompt = f"What number comes next in this sequence? {shown}, ?"
    return Task("sequence", diff, prompt, str(nxt), _exact(str(nxt)))


def _deduction(rng: random.Random, diff: int) -> Task:
    names = rng.sample(["Alice", "Bob", "Carol", "Dave", "Erin", "Frank"], 3)
    items = rng.sample(["red", "blue", "green", "gold", "silver"], 3)
    mapping = dict(zip(names, items))
    a, b, c = names
    clues = [
        f"{a} does not have {mapping[b]} or {mapping[c]}.",
        f"{b} has {mapping[b]}.",
    ]
    rng.shuffle(clues)
    prompt = (
        f"{a}, {b}, and {c} each hold a different colored card "
        f"({', '.join(items)}).\n" + " ".join(clues) +
        f"\nWhat color does {a} hold?"
    )
    return Task("deduction", diff, prompt, mapping[a], _exact(mapping[a]))


def _unscramble(rng: random.Random, diff: int) -> Task:
    words = {
        1: ["cable", "wires", "panel", "vault", "scan"],
        2: ["reactor", "shields", "engine", "oxygen", "sensor"],
        3: ["navigation", "calibrate", "diagnostic", "telemetry", "asteroid"],
    }[diff]
    word = rng.choice(words)
    scrambled = list(word)
    while "".join(scrambled) == word:
        rng.shuffle(scrambled)
    prompt = f"Unscramble these letters into a single English word: {' '.join(scrambled)}"
    return Task("unscramble", diff, prompt, word, _exact(word))


def _code_trace(rng: random.Random, diff: int) -> Task:
    a = rng.randint(2, 5)
    b = rng.randint(2, 6)
    loops = 2 + diff
    code = (
        "x = {a}\n"
        "for i in range(1, {loops}):\n"
        "    x = x * i + {b}\n"
        "print(x)"
    ).format(a=a, loops=loops + 1, b=b)
    x = a
    for i in range(1, loops + 1):
        x = x * i + b
    prompt = f"What does this Python program print?\n{code}"
    return Task("code_trace", diff, prompt, str(x), _exact(str(x)))


def _caesar(rng: random.Random, diff: int) -> Task:
    words = ["impostor", "emergency", "spaceship", "sabotage", "airlock", "venting"]
    word = rng.choice(words)
    shift = rng.randint(1, 5)
    enc = "".join(
        string.ascii_lowercase[(string.ascii_lowercase.index(ch) + shift) % 26]
        for ch in word
    )
    prompt = (
        f"This word was encrypted with a Caesar cipher (each letter shifted "
        f"forward by {shift}). Decrypt it: {enc}"
    )
    return Task("cipher", diff, prompt, word, _exact(word))


GENERATORS: dict[str, Callable[[random.Random, int], Task]] = {
    "arithmetic": _arithmetic,
    "sequence": _sequence,
    "deduction": _deduction,
    "unscramble": _unscramble,
    "code_trace": _code_trace,
    "cipher": _caesar,
}


class TaskBank:
    """Generates a fresh, varied set of capability tasks for one player."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    def draw(self, count: int) -> list[Task]:
        cats = list(GENERATORS)
        self.rng.shuffle(cats)
        tasks: list[Task] = []
        for i in range(count):
            cat = cats[i % len(cats)]
            diff = 1 + (i % 3)
            tasks.append(GENERATORS[cat](self.rng, diff))
        return tasks
