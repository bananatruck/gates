"""The feedback loop, held by test.

``tests/test_gate1.py`` holds each check in isolation. This file holds the thing
they compose into: engineer submits, gate rejects, feedback goes back, budget
advances by one agent turn, and the phase ends in the state the design says it
should. Every scenario in ``rig/scenarios.py`` is asserted against what it
documents, so a scenario and its description cannot drift apart.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gates import Ledger  # noqa: E402
from rig.loop import ScriptedEngineer, check_expectations, run_loop  # noqa: E402
from rig.reward import (  # noqa: E402
    LEGACY_MARKER,
    channel_sweep,
    legacy_view,
    upstream_detects_failure,
)
from rig.scenarios import SCENARIOS, CRASH_AFTER_HEAVY_OUTPUT  # noqa: E402


@pytest.fixture(scope="module")
def played(tmp_path_factory):
    """Play every scenario once; the assertions below read the results."""
    root = tmp_path_factory.mktemp("loop")
    return {
        name: run_loop(scenario, workdir=root / name, timeout_s=60)
        for name, scenario in SCENARIOS.items()
    }


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_behaves_as_documented(name, played):
    problems = check_expectations(SCENARIOS[name], played[name])
    assert not problems, "\n".join(problems)


# --------------------------------------------------------------------------- #
# the loop's own invariants
# --------------------------------------------------------------------------- #


def test_exhausted_budget_with_no_pass_produces_no_paper(played):
    """The terminal state that matters: a run that never worked cannot be written up."""
    outcome = played["archived-run"]
    assert outcome.outcome == "gate_failure"
    assert outcome.gate_failure is not None
    assert outcome.evidence_bundle is None
    assert outcome.turns_used == 3


def test_budget_counts_agent_turns_not_executions(played):
    """Three executions inside one turn must not spend three of the agent's rewrites."""
    outcome = played["inner-repair"]
    assert outcome.turns_used == 1
    assert len(outcome.executions) == 3
    assert outcome.turns[0].rejections_after == 0
    assert outcome.outcome == "pass"


def test_a_passing_turn_clears_the_rejection_count(played):
    outcome = played["recovers"]
    assert [t.rejections_after for t in outcome.turns] == [1, 2, 0]


def test_warnings_do_not_block_and_reach_the_writer(played):
    outcome = played["warn-tier"]
    execution = outcome.executions[0]
    assert execution.passed
    assert len(execution.warned_check_ids) >= 4
    assert "WARNINGS" in outcome.evidence_bundle


def test_rejection_feedback_names_a_fix(played):
    """Every rejection must hand the agent something it can act on."""
    for outcome in played.values():
        for execution in outcome.executions:
            if not execution.passed:
                assert "REQUIRED FIXES" in execution.feedback, execution.label


def test_every_execution_lands_in_the_ledger(played):
    for name, outcome in played.items():
        rows = Ledger(outcome.ledger_path).rows()
        assert len(rows) == len(outcome.executions), name
        assert {r["scenario"] for r in rows} == {name}
        for row, execution in zip(rows, outcome.executions):
            assert row["verdict"] == execution.report.verdict.value
            assert set(row["failed_checks"]) == execution.failed_check_ids
            assert set(row["warned_checks"]) == execution.warned_check_ids


def test_ledger_records_no_reward_score_without_a_reward_model(played):
    """Absent a real model, the reward column stays null rather than invented."""
    rows = Ledger(played["archived-run"].ledger_path).rows()
    assert all(row["reward_score"] is None for row in rows)


def test_a_reward_model_can_be_plugged_in(tmp_path):
    """The counterfactual arm: score the 1000-character view, not the real one."""
    seen = []

    def reward(view: str) -> float:
        seen.append(view)
        return 0.0 if LEGACY_MARKER in view else 1.0

    outcome = run_loop(
        SCENARIOS["archived-run"],
        workdir=tmp_path,
        reward_fn=reward,
        timeout_s=60,
    )
    rows = Ledger(outcome.ledger_path).rows()
    assert [r["reward_score"] for r in rows] == [1.0, 1.0, 1.0]
    assert all(len(v) <= 1000 for v in seen)


# --------------------------------------------------------------------------- #
# the reconstructed upstream channel
# --------------------------------------------------------------------------- #


def test_upstream_detector_is_blind_to_a_crash_behind_heavy_output(tmp_path):
    from gates import run_experiment

    execution = run_experiment(CRASH_AFTER_HEAVY_OUTPUT, tmp_path, timeout_s=60)
    assert execution.exception is not None
    assert not upstream_detects_failure(execution, max_len=1000)
    assert upstream_detects_failure(execution, max_len=100_000)
    assert LEGACY_MARKER not in legacy_view(execution, 1000)


def test_channel_sweep_finds_the_width_where_the_detector_starts_working(tmp_path):
    from gates import run_experiment

    execution = run_experiment(CRASH_AFTER_HEAVY_OUTPUT, tmp_path, timeout_s=60)
    points = channel_sweep(execution)
    visible = [p.max_len for p in points if p.marker_visible]
    assert 1_000 not in visible
    assert max(p.max_len for p in points) in visible
    # monotone: once the marker fits, wider channels keep it
    first = min(visible)
    assert all(p.marker_visible for p in points if p.max_len >= first)


def test_upstream_never_sees_a_missing_contract(tmp_path):
    """The saved run's shape: exits 0, prints numbers, records none.

    There is no marker to find at any channel width, so no amount of widening
    would have saved upstream here. Only the contract does.
    """
    from gates import run_experiment
    from rig.scenarios import NO_CONTRACT

    execution = run_experiment(NO_CONTRACT, tmp_path, timeout_s=60)
    assert execution.exit_code == 0
    assert not any(p.marker_visible for p in channel_sweep(execution))


# --------------------------------------------------------------------------- #
# the engineer seam
# --------------------------------------------------------------------------- #


def test_engineer_receives_the_feedback_report(tmp_path):
    """A real engineer is a model reading this text. It must actually arrive."""
    scenario = SCENARIOS["recovers"]
    engineer = ScriptedEngineer(scenario)
    run_loop(scenario, workdir=tmp_path, engineer=engineer, timeout_s=60)
    assert len(engineer.feedback_seen) == 2
    assert "static.no_unbound_names" in engineer.feedback_seen[0]
    assert "results.values_computed" in engineer.feedback_seen[1]


def test_loop_stops_when_the_engineer_gives_up(tmp_path):
    class GivesUp:
        def turn(self, feedback, turn_index):
            return None

    outcome = run_loop(
        SCENARIOS["archived-run"], workdir=tmp_path, engineer=GivesUp(), timeout_s=60
    )
    assert outcome.turns_used == 0
    assert outcome.outcome == "no_pass"
    assert outcome.gate_failure is None
