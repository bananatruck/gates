"""What the LLM layer costs, held to a ceiling.

Gate 1's standing claim is that a verdict costs zero model calls and that
rejection is cheap — the static tier exists so a broken program costs no compute
at all. The layer does not change the first, and this file stops it wrecking the
second. A prompt change that doubles what the gate costs should fail a test, not
show up on a bill.

Sizes are in characters rather than tokens, because `gates` has no tokenizer and
will not grow a dependency for one. Roughly four characters per token is close
enough for a ceiling.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gates import Gate1Config, run_gate1  # noqa: E402
from rig.loop import run_loop  # noqa: E402
from rig.scenarios import SCENARIOS  # noqa: E402

#: At most this many model calls per execution. The layer has two jobs and a
#: rejection is the only time both fire.
MAX_CALLS_PER_EXECUTION = 2

#: Prompt characters per execution, across every scenario. The log scan sends
#: the unflagged lines and the generator sends the report's facts; neither
#: should ever send a whole capture.
MAX_PROMPT_CHARS_PER_EXECUTION = 30_000


def stub(prompt, system):
    """Answers both jobs plausibly: an empty finding list, and a grounded fix."""
    if "JSON array" in system:
        return "[]"
    return "1. Bind every name before it is read."


@pytest.fixture
def config(tmp_path):
    def _make(**kw):
        kw.setdefault("timeout_s", 30)
        return Gate1Config(artifact_root=str(tmp_path), **kw)

    return _make


# --------------------------------------------------------------------------- #
# per-run accounting
# --------------------------------------------------------------------------- #


def test_a_passing_run_spends_at_most_one_call(config):
    """Nothing to fix, so the generator is never asked."""
    src = (
        "print('epoch 0 loss 1.2')\n"
        "record_metadata('seed', 0)\n"
        "v = 0.8\n"
        "record_result('a', v * 1.0)\n"
    )
    report = run_gate1(src, config(consult_model=stub))
    assert report.passed
    assert report.model["calls"] <= 1


def test_a_statically_rejected_run_never_reaches_the_log_scan(config):
    """The cheapest rejection there is stays cheap: nothing ran, so there are no
    logs to read, and only the generator is asked."""
    report = run_gate1("def f():\n    return missing_name\n", config(consult_model=stub))
    assert not report.passed
    assert report.execution is None
    assert report.model["calls"] == 1


def test_prompts_never_carry_a_whole_capture(config):
    """A run printing megabytes must not turn into a megabyte prompt."""
    sizes = []

    def measure(prompt, system):
        sizes.append(len(prompt))
        return "[]" if "JSON array" in system else "1. Fix it."

    src = "for i in range(4000):\n    print('x' * 200)\nraise RuntimeError('boom')\n"
    report = run_gate1(src, config(consult_model=measure))
    assert report.execution.stdout_bytes > 500_000
    assert sizes, "the model should have been consulted"
    assert max(sizes) <= Gate1Config().max_prompt_chars + 500


# --------------------------------------------------------------------------- #
# across the whole loop
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_stays_within_the_cost_ceiling(name, tmp_path):
    outcome = run_loop(
        SCENARIOS[name],
        workdir=tmp_path / name,
        timeout_s=60,
        consult_model=stub,
        counterfactual=False,
    )
    cost = outcome.model_cost
    executions = len(outcome.executions)
    assert cost["calls"] <= MAX_CALLS_PER_EXECUTION * executions, cost
    assert cost["prompt_chars"] <= MAX_PROMPT_CHARS_PER_EXECUTION * executions, cost
    assert cost["failures"] == 0, cost


def test_the_loop_still_behaves_identically_with_a_model(tmp_path):
    """The layer must change what the engineer reads, not what the gate decides."""
    from rig.loop import check_expectations

    for name, scenario in SCENARIOS.items():
        outcome = run_loop(
            scenario,
            workdir=tmp_path / name,
            timeout_s=60,
            consult_model=stub,
            counterfactual=False,
        )
        assert check_expectations(scenario, outcome) == [], name


# --------------------------------------------------------------------------- #
# the failure policy
# --------------------------------------------------------------------------- #


def test_an_unavailable_model_does_not_stall_or_fail_the_loop(tmp_path):
    """A validity layer that becomes unavailable when an API is down is not a
    validity layer."""
    from rig.loop import check_expectations

    def dead(prompt, system):
        raise ConnectionError("provider unreachable")

    scenario = SCENARIOS["archived-run"]
    outcome = run_loop(
        scenario, workdir=tmp_path, timeout_s=60, consult_model=dead,
        counterfactual=False,
    )
    assert check_expectations(scenario, outcome) == []
    assert outcome.outcome == "gate_failure"
    assert outcome.model_cost["degraded"]
    for execution in outcome.executions:
        assert "REQUIRED FIXES" in execution.feedback


def test_a_slow_model_cannot_hold_the_gate_open(config):
    import time

    def slow(prompt, system):
        time.sleep(20)
        return "1. Too late."

    started = time.monotonic()
    report = run_gate1(
        "def f():\n    return missing_name\n",
        config(consult_model=slow, model_timeout_s=0.5),
    )
    elapsed = time.monotonic() - started
    assert not report.passed
    assert elapsed < 10, "the model deadline did not bound the gate"
    assert report.model_degraded
