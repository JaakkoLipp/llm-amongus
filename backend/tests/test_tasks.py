import random

from app.game.tasks import GENERATORS, TaskBank, extract_answer


def test_every_generator_self_consistent():
    """A task's own expected answer must pass its own checker, at all difficulties."""
    rng = random.Random(123)
    for name, gen in GENERATORS.items():
        for diff in (1, 2, 3):
            task = gen(rng, diff)
            assert task.category == name
            assert task.check(f"ANSWER: {task.expected}"), f"{name}/{diff} failed self-check"
            # A clearly wrong answer must not pass.
            assert not task.check("ANSWER: definitely-not-it")


def test_extract_answer_prefers_marker():
    assert extract_answer("reasoning here\nANSWER: 42") == "42"
    assert extract_answer("no marker\nfinal line") == "final line"
    assert extract_answer("") == ""


def test_taskbank_draws_requested_count():
    bank = TaskBank(random.Random(1))
    tasks = bank.draw(5)
    assert len(tasks) == 5
