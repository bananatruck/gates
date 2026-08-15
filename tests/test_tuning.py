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


# --------------------------------------------------------------------------- #
# the ablation: Gate 1 on vs Gate 1 off
# --------------------------------------------------------------------------- #

CRASH_AFTER_OUTPUT = (
    "for i in range(60):\n"
    "    print(f'epoch {i:03d}  loss 1.9243  train_acc 0.4120  val_acc 0.3980')\n"
    "acc = correct / total\n"
)
RECORDS = (
    "import random\nrandom.seed(0)\nrecord_metadata('seed', 0)\n"
    "for i in range(60):\n    print(f'epoch {i} loss 1.0')\n"
    "acc = 408/500\nrecord_result('exp1.test_acc', acc, unit='ratio')\n"
)


def scripted(*replies):
    state = {"i": 0}

    def fn(prompt, system):
        i = min(state["i"], len(replies) - 1)
        state["i"] += 1
        return replies[i]

    return fn


def test_the_ungated_arm_accepts_a_crash_that_printed_past_the_ceiling(tmp_path):
    """The archived run's shape, reproduced against the real runner.

    The crash marker is appended after the program's own output and the view is
    sliced to 1,000 characters, so upstream's only failure test never fires.
    """
    from rig.ablation import run_ungated

    arm = run_ungated(
        scripted(CRASH_AFTER_OUTPUT), "task", tmp_path, max_turns=2, timeout_s=60
    )
    assert arm.accepted
    assert arm.accepted_at == 1
    assert arm.accepted_a_crash == 1
    assert arm.accepted_without_results == 1
    assert arm.false_success == 1
    assert arm.attempts[0].exit_code == 1
    assert arm.attempts[0].stdout_bytes > 1000


def test_the_gated_arm_rejects_it_and_the_engineer_recovers(tmp_path):
    from rig.ablation import run_gated

    arm = run_gated(
        scripted(CRASH_AFTER_OUTPUT, RECORDS),
        "task",
        ("exp1.test_acc",),
        tmp_path,
        max_turns=3,
        timeout_s=60,
    )
    assert arm.accepted
    assert arm.accepted_at == 2
    assert arm.false_success == 0
    assert arm.accepted_a_crash == 0
    assert "exec.no_uncaught_exception" in arm.attempts[0].failed_checks
    assert arm.attempts[1].recorded == {"exp1.test_acc": pytest.approx(0.816)}


def test_a_short_crash_is_caught_by_both_arms(tmp_path):
    """The divergence is conditional, not universal.

    A run that dies before printing 1,000 characters leaves the marker inside
    the slice, and upstream catches it too. Holding this stops the ablation
    being read as a blanket claim.
    """
    from rig.ablation import run_ungated

    arm = run_ungated(
        scripted("acc = missing_name / 2\n"), "task", tmp_path, max_turns=1, timeout_s=60
    )
    assert not arm.accepted
    assert arm.false_success == 0


# --------------------------------------------------------------------------- #
# which local model can hold which role — measured, not assumed
# --------------------------------------------------------------------------- #

#: Measured 2026-08-15 against a local Ollama on an RTX 4060.
#:
#: The gate's two jobs emit short structured output and qwen3:8b handles them at
#: ~16s a call. The engineer role is different in kind: it must emit a whole
#: program, and a reasoning model at this scale spends its entire token budget in
#: the reasoning channel and returns finish_reason="length" with EMPTY content.
#: That was reproduced at 2,500, 3,000, 6,000 and 8,000 token ceilings, with and
#: without "/no_think", with think=False, and with
#: chat_template_kwargs={"enable_thinking": False} -- none of which suppressed it
#: through Ollama's OpenAI-compatible endpoint.
#:
#: A code model without a reasoning mode writes the same program in 42s.
LOCAL_MODEL_ROLES = {
    "qwen3:8b": {
        "gate_jobs": "suitable — ~16s per call, 97-118 chars out",
        "engineer": "unsuitable — empty content at every ceiling tried",
        "empty_content_ceilings_tried": [2500, 3000, 6000, 8000],
        "thinking_switches_tried": ["/no_think", "think=False",
                                    "chat_template_kwargs.enable_thinking"],
    },
    "qwen2.5-coder:7b": {
        "engineer": "suitable — 2,396 chars and 3 record_result calls in 42s",
    },
}


def test_the_engineer_and_gate_roles_have_different_model_requirements():
    """Recorded so the split is a measurement rather than a preference.

    It also explains why the ablation runs two local models: the gates stay on
    qwen3:8b, and only the engineer moved.
    """
    q3 = LOCAL_MODEL_ROLES["qwen3:8b"]
    assert "suitable" in q3["gate_jobs"]
    assert "unsuitable" in q3["engineer"]
    # the failure was not a ceiling that could simply be raised
    assert max(q3["empty_content_ceilings_tried"]) >= 8000
    assert len(q3["thinking_switches_tried"]) == 3
    assert "suitable" in LOCAL_MODEL_ROLES["qwen2.5-coder:7b"]["engineer"]
