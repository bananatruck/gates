"""Running the host exactly as shipped, so the comparison has a baseline.

A validity layer that cannot be switched off cannot be evaluated. The obvious
alternative -- check out the pre-Gate-1 branch and run that -- changes the model
plumbing, the rate-limit backoff and the prompts along with the gate, so any
difference between the two papers could be pinned on those instead. The bypass
exists to make the gate the only difference.

Which means the bypass has to be *faithful*, not merely permissive. Upstream
appended its crash marker after the program's own output and then sliced the
whole buffer to 1,000 characters, so a run that printed past the ceiling lost
the marker and its crash became invisible. A bypass that tidied that up would
quietly hand the baseline a better channel than it has, and every measured gap
would be understated. These tests pin the defect in place.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gates.adapters.agentlab import (  # noqa: E402
    LEGACY_MAX_LEN,
    GatedExecution,
    gate1_enabled,
    gated_execute,
    make_context,
)


@pytest.fixture
def gate_off(monkeypatch):
    monkeypatch.setenv("GATES_GATE1", "off")


# --------------------------------------------------------------------------- #
# the switch
# --------------------------------------------------------------------------- #


def test_gate_is_on_unless_explicitly_disabled(monkeypatch):
    monkeypatch.delenv("GATES_GATE1", raising=False)
    assert gate1_enabled()


@pytest.mark.parametrize("value", ["off", "OFF", "0", "false", "no", " off "])
def test_recognised_off_values(monkeypatch, value):
    monkeypatch.setenv("GATES_GATE1", value)
    assert not gate1_enabled()


def test_an_unrecognised_value_leaves_the_gate_on(monkeypatch):
    """Failing open here would silently disable the layer under a typo."""
    monkeypatch.setenv("GATES_GATE1", "disabled-ish")
    assert gate1_enabled()


# --------------------------------------------------------------------------- #
# the acceptance rule, reproduced rather than approximated
# --------------------------------------------------------------------------- #


def test_a_clean_run_is_accepted(tmp_path, gate_off):
    ctx = make_context(research_dir=str(tmp_path))
    outcome = gated_execute("print('hello')", ctx)
    assert outcome.passed
    assert outcome.report is None
    assert "hello" in outcome.evidence_bundle


def test_a_short_crash_is_caught_because_the_marker_survives(tmp_path, gate_off):
    ctx = make_context(research_dir=str(tmp_path))
    outcome = gated_execute("raise ValueError('boom')", ctx)
    assert not outcome.passed
    assert "[CODE EXECUTION ERROR]" in outcome.evidence_bundle


def test_a_noisy_crash_is_accepted_because_the_marker_falls_off(tmp_path, gate_off):
    """The defect, reproduced: print past the ceiling, then die, and pass.

    This is the single most important test in the file. If the bypass ever
    starts catching this, the baseline has been silently improved and every
    comparison drawn against it is too flattering to Gate 1.
    """
    code = f"print('x' * {LEGACY_MAX_LEN * 2})\nraise ValueError('boom')"
    ctx = make_context(research_dir=str(tmp_path))
    outcome = gated_execute(code, ctx)
    assert outcome.passed, "upstream accepted this, so the bypass must too"
    assert "[CODE EXECUTION ERROR]" not in outcome.evidence_bundle


def test_the_channel_is_capped_at_the_upstream_ceiling(tmp_path, gate_off):
    ctx = make_context(research_dir=str(tmp_path))
    outcome = gated_execute(f"print('y' * {LEGACY_MAX_LEN * 3})", ctx)
    assert len(outcome.evidence_bundle) == LEGACY_MAX_LEN


def test_results_recorded_by_the_experiment_are_not_surfaced(tmp_path, gate_off):
    """No registry on the bypass path — that absence is the finding."""
    ctx = make_context(research_dir=str(tmp_path))
    outcome = gated_execute("print('done')", ctx)
    assert outcome.report is None
    assert "VERIFIED RESULTS" not in outcome.evidence_bundle


# --------------------------------------------------------------------------- #
# the wrapper
# --------------------------------------------------------------------------- #


def test_passed_does_not_crash_without_a_report():
    """`passed` used to dereference `report` unconditionally."""
    assert GatedExecution(report=None, feedback="", evidence_bundle="",
                          ungated_passed=True).passed
    assert not GatedExecution(report=None, feedback="", evidence_bundle="",
                              ungated_passed=False).passed


def test_summary_is_available_without_a_report():
    out = GatedExecution(report=None, feedback="", evidence_bundle="",
                         ungated_passed=True)
    assert "bypassed" in out.summary


def test_gate_on_still_produces_a_report(tmp_path, monkeypatch):
    monkeypatch.delenv("GATES_GATE1", raising=False)
    ctx = make_context(research_dir=str(tmp_path))
    outcome = gated_execute("print('hello')", ctx)
    assert outcome.report is not None
