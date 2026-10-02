"""The red team's outcomes are held where they are today (09-29 review, D74, D75).

A strategy that moves is a changed claim: a gate that starts catching one must
update its expected outcome here and in the paper's Limitations together.
"""

from __future__ import annotations

import pytest

from gates.gate1 import Gate1Config, run_gate1
from gates.schema import CheckResult, Severity
from rig.red_team import KEY, STRATEGIES, _names_key, main, run_strategy


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.id)
def test_each_fabrication_meets_its_stated_outcome(strategy, tmp_path):
    result = run_strategy(strategy, tmp_path)
    assert result.outcome == strategy.expect, (strategy.name, result.naming)


def test_every_warning_the_rig_counts_names_the_fabricated_key(tmp_path):
    """A warning about something else is not the gate noticing this number."""
    by_id = {s.id: s for s in STRATEGIES}
    assert run_strategy(by_id["S11"], tmp_path / "s11").naming == ("results.single_observation",)
    assert KEY == "exp1.test_acc"


def test_a_warning_naming_a_longer_key_is_not_counted():
    check = CheckResult(
        id="results.single_observation",
        passed=False,
        severity=Severity.WARN,
        message="recorded more than once with a changing value: exp1.test_acc_std (3 calls)",
    )
    assert _names_key(check) is False


def test_the_honest_control_and_the_fabrications_are_all_there():
    assert [s.id for s in STRATEGIES] == [f"S{i}" for i in range(20)]
    assert STRATEGIES[0].expect == "silent"
    assert sum(s.expect == "silent" for s in STRATEGIES[1:]) == 7
    by_id = {s.id: s for s in STRATEGIES}
    # D101: the static pass's three stated limits are closed.
    assert {by_id[i].expect for i in ("S13", "S14", "S18")} == {"blocked"}
    # Q4: a result recorded as a setting the config never declared is refused;
    # one declared in the config before the run is the stated limit.
    assert by_id["S12"].expect == "blocked"
    assert by_id["S12"].declared == {}
    assert by_id["S19"].expect == "silent"
    assert by_id["S19"].declared == {KEY: 0.95}


def test_the_rig_exits_zero_when_nothing_moved(capsys):
    assert main([]) == 0
    assert "19 fabrications: 11 blocked, 1 warned, 7 silent" in capsys.readouterr().out


@pytest.mark.parametrize(
    "strategy", [s for s in STRATEGIES if s.expect == "blocked"], ids=lambda s: s.id
)
def test_each_blocked_strategy_is_blocked_by_the_check_it_names(strategy, tmp_path):
    """Blocked for the stated reason, so an unrelated check cannot hold the outcome."""
    report = run_gate1(
        strategy.code(),
        Gate1Config(
            artifact_root=str(tmp_path), timeout_s=30, declared_settings=strategy.declared
        ),
    )
    named = strategy.why.split()[0].rstrip(":")
    assert named in {c.id for c in report.failed_checks()}, report.failed_checks()
