"""The setup-time budget prompt.

The point of this module is a sentence someone reads before a run costs them
money, so the tests check that the sentence is shown and that a hurried answer
lands somewhere sane rather than that the arithmetic works.
"""

from __future__ import annotations

import pytest

from gates.gate1 import Gate1Config
from gates.gate2 import Gate2Config
from gates.gate3 import Gate3Config
from gates.setup import WARNING, Budgets, defaults, prompt_budgets


def answering(*replies):
    """A fake terminal: each prompt gets the next reply, then empty lines."""
    queue = list(replies)
    shown: list[str] = []

    def ask(prompt):
        shown.append(prompt)
        return queue.pop(0) if queue else ""

    ask.shown = shown
    return ask


def test_the_cost_warning_is_shown_before_anything_is_asked():
    shown: list[str] = []
    prompt_budgets(ask=answering(), show=shown.append)
    assert "can increase the cost of a run" in shown[0]
    assert "Leave a field empty to accept our default" in shown[0]


def test_an_empty_field_accepts_our_default():
    assert prompt_budgets(ask=answering("", "", ""), show=lambda _: None) == defaults()


def test_a_number_defines_that_gate_only():
    budgets = prompt_budgets(ask=answering("5", "", "1"), show=lambda _: None)
    assert budgets == Budgets(gate1=5, gate2=defaults().gate2, gate3=1)


def test_the_defaults_are_the_gates_own():
    """One source of truth. A default that disagrees with its gate is worse
    than no default."""
    assert defaults() == Budgets(
        gate1=Gate1Config().max_attempts,
        gate2=Gate2Config().max_attempts,
        gate3=Gate3Config().max_attempts,
    )


@pytest.mark.parametrize("bad", ["two", "0", "-1", "3.5"])
def test_an_unusable_answer_is_refused_and_re_asked(bad):
    said: list[str] = []
    budgets = prompt_budgets(ask=answering(bad, "4"), show=said.append)
    assert budgets.gate1 == 4
    assert any("Leave empty for" in line for line in said)


def test_end_of_input_accepts_the_defaults_rather_than_crashing():
    """A setup script reading from a closed pipe should configure the run the
    way we recommend, not take down the host installing us."""

    def ask(_):
        raise EOFError

    assert prompt_budgets(ask=ask, show=lambda _: None) == defaults()


def test_the_warning_names_every_gate():
    assert "gates 1, 2 and 3" in WARNING
