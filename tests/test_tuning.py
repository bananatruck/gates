"""The arm-comparison harness, exercised with stub engineers.

What is held here is that the harness measures the right thing: that the two
arms differ in exactly one variable, that turns-to-pass is counted correctly,
and that a run which never converges is recorded as stuck rather than dropped.

What is *not* here is the comparison itself. That needs a real engineer model
and an API key, and inventing numbers for it would be the failure this whole
project is about.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rig.tuning import (  # noqa: E402
    TASKS,
    ArmResult,
    ModelEngineer,
    render_comparison,
    run_arm,
)

GOOD = """\
import random
seed = 0
random.seed(seed)
record_metadata("seed", seed)
correct = sum(1 for _ in range(500) if random.random() < 0.8)
acc = correct / 500
record_result("exp1.test_acc", acc, unit="ratio")
"""

BROKEN = "acc = correct_count / total\nrecord_result('exp1.test_acc', acc)\n"


def engineer_that(*responses):
    """A stub engineer replying with each response in turn."""
    state = {"i": 0}

    def fn(prompt, system):
        i = min(state["i"], len(responses) - 1)
        state["i"] += 1
        return responses[i]

    return fn


ACCURACY = TASKS[0]


# --------------------------------------------------------------------------- #
# the engineer seam
# --------------------------------------------------------------------------- #


def test_the_engineer_is_told_the_task_and_the_contract():
    engineer = ModelEngineer(engineer_that(GOOD), ACCURACY)
    engineer.turn(None, 0)
    prompt = engineer.prompts[0]
    assert ACCURACY.brief in prompt
    assert "record_result" in prompt
    assert "REJECTED" not in prompt


def test_a_rewrite_carries_the_feedback_report():
    engineer = ModelEngineer(engineer_that(GOOD), ACCURACY)
    engineer.turn(None, 0)
    engineer.turn("GATE 1: FAIL\n\n  [static.no_unbound_names]\n", 1)
    assert "REJECTED" in engineer.prompts[1]
    assert "static.no_unbound_names" in engineer.prompts[1]


def test_code_fences_are_stripped():
    engineer = ModelEngineer(engineer_that(f"```python\n{GOOD}```"), ACCURACY)
    steps = engineer.turn(None, 0)
    assert steps[0].code.startswith("import random")
    assert "```" not in steps[0].code


def test_an_exploding_engineer_ends_the_loop_rather_than_the_process():
    def explode(prompt, system):
        raise RuntimeError("provider down")

    assert ModelEngineer(explode, ACCURACY).turn(None, 0) is None


def test_an_empty_reply_ends_the_loop():
    assert ModelEngineer(engineer_that("   "), ACCURACY).turn(None, 0) is None


def test_the_engineer_stops_at_the_budget():
    engineer = ModelEngineer(engineer_that(GOOD), ACCURACY, max_turns=2)
    assert engineer.turn(None, 0) is not None
    assert engineer.turn("f", 1) is not None
    assert engineer.turn("f", 2) is None


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #


def test_an_arm_that_passes_first_time_records_one_turn(tmp_path):
    result = run_arm(
        engineer_that(GOOD),
        generated_feedback=False,
        workdir=tmp_path,
        tasks=(ACCURACY,),
        seeds=1,
    )
    assert result.pass_rate == 1.0
    assert result.turns_to_pass == [1]
    assert result.stuck_on() == {}


def test_an_arm_that_recovers_on_the_second_turn_records_two(tmp_path):
    result = run_arm(
        engineer_that(BROKEN, GOOD),
        generated_feedback=False,
        workdir=tmp_path,
        tasks=(ACCURACY,),
        seeds=1,
    )
    assert result.pass_rate == 1.0
    assert result.turns_to_pass == [2]


def test_a_run_that_never_converges_is_recorded_as_stuck(tmp_path):
    result = run_arm(
        engineer_that(BROKEN),
        generated_feedback=False,
        workdir=tmp_path,
        tasks=(ACCURACY,),
        seeds=1,
    )
    assert result.pass_rate == 0.0
    assert result.mean_turns is None
    stuck = result.stuck_on()
    assert stuck, "a failed run must name what it was stuck on"
    assert any("results" in k or "exec" in k or "static" in k for k in stuck)


def test_the_template_arm_withholds_the_model_from_the_gate(tmp_path):
    """Arm A must actually produce template feedback, or the arms are the same."""
    result = run_arm(
        engineer_that(BROKEN),
        generated_feedback=False,
        workdir=tmp_path,
        tasks=(ACCURACY,),
        seeds=1,
    )
    assert all(run["model_calls"] == 0 for run in result.runs)


def test_the_generated_arm_supplies_it(tmp_path):
    result = run_arm(
        engineer_that(BROKEN),
        generated_feedback=True,
        workdir=tmp_path,
        tasks=(ACCURACY,),
        seeds=1,
    )
    assert sum(run["model_calls"] for run in result.runs) > 0


def test_seeds_multiply_the_runs(tmp_path):
    result = run_arm(
        engineer_that(GOOD),
        generated_feedback=False,
        workdir=tmp_path,
        tasks=(ACCURACY,),
        seeds=3,
    )
    assert len(result.runs) == 3


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def test_the_comparison_states_the_cost_confound():
    template = ArmResult(arm="template")
    generated = ArmResult(arm="generated")
    text = render_comparison(template, generated)
    assert "extra model call per rejection" in text
    assert "Report both or neither" in text


def test_every_task_declares_keys_the_gate_will_enforce():
    for task in TASKS:
        assert task.expected_keys
        assert all("." in key for key in task.expected_keys)
