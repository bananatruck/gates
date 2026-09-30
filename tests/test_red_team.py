"""The red team's outcomes are held where they are today (09-29 review, D74, D75).

A strategy that moves is a changed claim: a gate that starts catching one must
update its expected outcome here and in the paper's Limitations together.
"""

from __future__ import annotations

import pytest

from gates.gate1 import Gate1Config, run_gate1
from rig.red_team import KEY, STRATEGIES, main, run_strategy


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.id)
def test_each_fabrication_meets_its_stated_outcome(strategy, tmp_path):
    result = run_strategy(strategy, tmp_path)
    assert result.outcome == strategy.expect, (strategy.name, result.naming)


def test_every_warning_the_rig_counts_names_the_fabricated_key(tmp_path):
    """A warning about something else is not the gate noticing this number."""
    by_id = {s.id: s for s in STRATEGIES}
    assert run_strategy(by_id["S11"], tmp_path / "s11").naming == ("results.single_observation",)
    assert KEY == "exp1.test_acc"


def test_the_honest_control_and_the_fabrications_are_all_there():
    assert [s.id for s in STRATEGIES] == [f"S{i}" for i in range(18)]
    assert STRATEGIES[0].expect == "silent"
    assert sum(s.expect == "silent" for s in STRATEGIES[1:]) == 9


def test_the_rig_exits_zero_when_nothing_moved(capsys):
    assert main([]) == 0
    assert "17 fabrications: 7 blocked, 1 warned, 9 silent" in capsys.readouterr().out


@pytest.mark.parametrize(
    "strategy", [s for s in STRATEGIES if s.expect == "blocked"], ids=lambda s: s.id
)
def test_each_blocked_strategy_is_blocked_by_the_check_it_names(strategy, tmp_path):
    """Blocked for the stated reason, so an unrelated check cannot hold the outcome."""
    report = run_gate1(strategy.code(), Gate1Config(artifact_root=str(tmp_path), timeout_s=30))
    named = strategy.why.split()[0].rstrip(":")
    assert named in {c.id for c in report.failed_checks()}, report.failed_checks()
