"""The generated feedback report, and the grounding that makes it safe.

The generator is the reason Gate 1 has an LLM layer. The grounding check is the
reason having one is not a new failure mode: a fix that names a variable the
code does not contain sends the ML engineer chasing something that does not
exist, which is the failure the gate exists to prevent, reintroduced at its exit.

So the tests that matter here are the adversarial ones. A model that writes a
beautiful, specific, entirely invented fix must be rejected whole.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gates import Gate1Config, ModelLayer, run_gate1  # noqa: E402
from gates.llm_report import (  # noqa: E402
    build_vocabulary,
    check_grounding,
    generate_fixes,
)
from gates.report import render_feedback  # noqa: E402

# The archived run's shape: a name read inside a method, bound nowhere.
UNBOUND = (
    "class GCN:\n"
    "    def __init__(self, in_dim):\n"
    "        self.in_dim = in_dim\n"
    "\n"
    "    def forward(self, x):\n"
    "        return [v * hidden_dim for v in x]\n"
    "\n"
    "model = GCN(1433)\n"
    "acc = sum(model.forward([0.5])) / 1\n"
    "record_result('exp1.acc', acc)\n"
)

NO_CONTRACT = "print('Final test accuracy: 0.8160')\n"

CLEAN = (
    "v = 408 / 500\n"
    "record_metadata('seed', 0)\n"
    "record_result('exp1.acc', v, unit='ratio')\n"
)


def model_returning(payload: str):
    return lambda prompt, system: payload


@pytest.fixture
def config(tmp_path):
    def _make(**kw):
        kw.setdefault("timeout_s", 30)
        return Gate1Config(artifact_root=str(tmp_path), **kw)

    return _make


@pytest.fixture
def unbound_report(config):
    """A real rejection to write fixes against."""
    return run_gate1(UNBOUND, config())


# --------------------------------------------------------------------------- #
# grounding
# --------------------------------------------------------------------------- #


def test_vocabulary_contains_the_names_the_run_actually_used(unbound_report):
    names, _ = build_vocabulary(unbound_report, UNBOUND)
    assert {"GCN", "forward", "hidden_dim", "in_dim", "record_result"} <= names


def test_a_grounded_fix_passes(unbound_report):
    text = (
        "1. Bind `hidden_dim` before it is read, or pass it into `GCN.__init__` "
        "and read it as `self.hidden_dim` inside `forward`."
    )
    assert check_grounding(text, unbound_report, UNBOUND) == []


def test_an_invented_variable_is_caught(unbound_report):
    text = "1. Set `embedding_width` before calling `build_projection_head`."
    offenders = check_grounding(text, unbound_report, UNBOUND)
    assert "embedding_width" in offenders
    assert "build_projection_head" in offenders


def test_an_invented_line_number_is_caught(unbound_report):
    text = "1. Bind `hidden_dim` at line 337 before it is read."
    assert "line 337" in check_grounding(text, unbound_report, UNBOUND)


def test_a_real_line_number_passes(unbound_report):
    text = "1. Bind `hidden_dim` before line 6, where it is read."
    assert check_grounding(text, unbound_report, UNBOUND) == []


def test_ordinary_prose_is_not_policed(unbound_report):
    """The rule constrains names and numbers, not English."""
    text = (
        "1. The propagation width must be established before the forward pass "
        "runs, otherwise nothing downstream can be trusted."
    )
    assert check_grounding(text, unbound_report, UNBOUND) == []


def test_call_syntax_inside_backticks_does_not_trip_the_check(unbound_report):
    text = '1. Call `record_result("exp1.acc", acc)` once the value is computed.'
    assert check_grounding(text, unbound_report, UNBOUND) == []


def test_offenders_are_reported_once_each(unbound_report):
    text = "1. Use `ghost_var`. 2. Then use `ghost_var` again with `ghost_var`."
    assert check_grounding(text, unbound_report, UNBOUND) == ["ghost_var"]


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #


def test_a_passing_run_asks_for_nothing(config):
    calls = []

    def count(prompt, system):
        calls.append(prompt)
        return "[]"

    report = run_gate1(CLEAN, config(consult_model=count))
    assert report.passed
    assert report.generated_fixes is None
    # Nothing printed, so no scan call either: a clean silent run is free.
    assert calls == []


def test_grounded_output_becomes_the_reports_fixes(config):
    fix = "1. Bind `hidden_dim` before it is read, or pass it into `GCN.__init__`."
    report = run_gate1(UNBOUND, config(consult_model=model_returning(fix)))
    assert not report.passed
    assert report.generated_fixes == fix
    text = render_feedback(report)
    assert "REQUIRED FIXES" in text
    assert "pass it into `GCN.__init__`" in text


def test_ungrounded_output_is_rejected_whole_and_the_template_renders(config):
    """Not repaired, not partially kept. The whole section is discarded."""
    invented = "1. Increase `dropout_rate` and re-run `train_with_warmup`."
    report = run_gate1(UNBOUND, config(consult_model=model_returning(invented)))
    assert report.generated_fixes is None
    text = render_feedback(report)
    assert "dropout_rate" not in text
    assert "train_with_warmup" not in text
    # the deterministic template took over
    assert "Bind every name listed above" in text
    # names the actual fault rather than reporting an outage that did not happen
    assert "dropout_rate" not in text
    assert "this run does not contain" in text
    assert "could not be reached" not in text


def test_the_rejection_is_recorded_rather_than_silent(config):
    invented = "1. Adjust `nonexistent_knob`."
    report = run_gate1(UNBOUND, config(consult_model=model_returning(invented)))
    check = next(c for c in report.checks if c.id == "report.fixes_grounded")
    assert "nonexistent_knob" in check.evidence["ungrounded"]
    assert check.passed  # informational: a rejected suggestion is not a run fault


def test_preamble_and_code_fences_are_stripped(config):
    noisy = (
        "Certainly! Here are the fixes you asked for:\n\n"
        "1. Bind `hidden_dim` before it is read.\n"
        "2. Record every metric with `record_result`.\n"
        "```\nirrelevant\n```\n"
    )
    report = run_gate1(UNBOUND, config(consult_model=model_returning(noisy)))
    assert report.generated_fixes.startswith("1. Bind")
    assert "Certainly" not in report.generated_fixes
    assert "irrelevant" not in report.generated_fixes


def test_findings_stay_deterministic_even_when_fixes_are_generated(config):
    """The model writes the fixes, not the findings."""
    lying = "1. Bind `hidden_dim` before it is read."
    report = run_gate1(UNBOUND, config(consult_model=model_returning(lying)))
    text = render_feedback(report)
    assert "[static.no_unbound_names]" in text
    assert "name(s) read but never bound: hidden_dim" in text


def test_no_model_falls_back_without_claiming_degradation(config):
    """A deployment with no model is not a degraded one; the template is simply
    what its report is."""
    report = run_gate1(UNBOUND, config())
    text = render_feedback(report)
    assert "REQUIRED FIXES" in text
    assert "could not be reached" not in text
    assert "standard guidance" not in text


def test_a_failing_model_degrades_and_says_so(config):
    def explode(prompt, system):
        raise TimeoutError("gateway timeout")

    report = run_gate1(UNBOUND, config(consult_model=explode))
    assert report.generated_fixes is None
    assert report.model_degraded
    text = render_feedback(report)
    assert "Bind every name listed above" in text
    assert "could not be reached" in text
    assert "does not contain" not in text


def test_generation_never_runs_before_the_verdict(config):
    """The prompt is allowed to contain the verdict — which is only possible
    because the verdict was already decided when the model was asked."""
    seen = {}

    def capture(prompt, system):
        seen["prompt"] = prompt
        return "1. Bind `hidden_dim` before it is read."

    run_gate1(UNBOUND, config(consult_model=capture))
    assert "VERDICT: FAIL" in seen["prompt"]


def test_missing_contract_gets_a_fix_naming_the_api(config):
    fix = '1. Record the accuracy with `record_result("exp1.acc", acc)`.'
    report = run_gate1(NO_CONTRACT, config(consult_model=model_returning(fix)))
    assert not report.passed
    assert report.generated_fixes == fix
    assert "record_result" in render_feedback(report)


def test_a_model_answering_with_nothing_falls_back(config):
    for empty in ("", "   ", "I have no suggestions."):
        report = run_gate1(UNBOUND, config(consult_model=model_returning(empty)))
        if report.generated_fixes:
            assert report.generated_fixes.strip()
        else:
            assert "Bind every name listed above" in render_feedback(report)


def test_a_proposed_key_is_not_treated_as_a_claim_about_existing_code(unbound_report):
    """`record_result("new.key", v)` proposes a key; it does not assert one.

    Grounding this against the current source would make the one fix
    results.contract_present needs impossible to write.
    """
    text = '1. Record it with `record_result("exp2.speedup", speedup)`.'
    assert check_grounding(text, unbound_report, UNBOUND) == []


def test_string_literals_are_not_checked_as_identifiers(unbound_report):
    text = '1. Print `"training complete"` when the loop ends.'
    assert check_grounding(text, unbound_report, UNBOUND) == []


def test_the_template_exemption_does_not_extend_to_other_calls(unbound_report):
    """Only the harness APIs are templates. Everything else still refers."""
    text = "1. Call `configure_optimiser(lr)` before training."
    assert "configure_optimiser" in check_grounding(text, unbound_report, UNBOUND)


# --------------------------------------------------------------------------- #
# found by the first live Gemini run
# --------------------------------------------------------------------------- #


def test_prose_in_backticks_is_not_read_as_code(unbound_report):
    """A live run rejected a good fix over ['so', 'that', 'it', 'when',
    'executing'] because the model emphasised a phrase with backticks."""
    text = (
        "1. Move the assignment up `so that it is bound when executing` the "
        "forward pass."
    )
    assert check_grounding(text, unbound_report, UNBOUND) == []


def test_code_like_spans_are_still_checked(unbound_report):
    """The prose exemption must not become a way to smuggle invented names."""
    assert "ghost_fn" in check_grounding(
        "1. Call `ghost_fn(x)` first.", unbound_report, UNBOUND
    )
    assert "ghost_var" in check_grounding(
        "1. Set `ghost_var`.", unbound_report, UNBOUND
    )


def test_artifact_paths_are_not_shown_to_the_model(config):
    """The model cannot cite a path it never sees.

    A live run produced "Define `n_classes` before line 20 in
    `/home/.../attempt_03/experiment.py`" -- grounded, because every segment of
    that path is in the evidence, and useless, because the engineer edits its
    own program and has never heard of the gate's copy.
    """
    seen = {}

    def capture(prompt, system):
        seen["prompt"] = prompt
        seen["system"] = system
        return "1. Bind `hidden_dim` before it is read."

    run_gate1(UNBOUND, config(consult_model=capture))
    assert "/attempt_" not in seen["prompt"]
    assert "stdout.txt" not in seen["prompt"]
    assert "experiment.py" not in seen["prompt"]
    assert "file path" in seen["system"]
