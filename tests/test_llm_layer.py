"""The LLM layer, and the guarantee that lets a required model coexist with a
deterministic verdict.

Gate 1 needs a model — the feedback report is the loop's return path to the ML
engineer, and no template writes it. R1.9 still says the verdict is computed with
no LLM in the path. Both hold because of *where* the layer sits and *what
severity* it can emit, and those are the two things this file pins down.
"""

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gates import (  # noqa: E402
    Gate1Config,
    ModelLayer,
    Severity,
    model_warning,
    run_gate1,
)
from gates import llm  # noqa: E402

LLM_SOURCE = Path(llm.__file__).read_text(encoding="utf-8")

CLEAN = (
    "correct = 408\n"
    "total = 500\n"
    "acc = correct / total\n"
    "record_metadata('seed', 0)\n"
    "record_result('exp1.acc', acc, unit='ratio')\n"
)

BROKEN = "def f(x):\n    return x * undefined_width\n"


@pytest.fixture
def config(tmp_path):
    def _make(**kw):
        kw.setdefault("timeout_s", 30)
        return Gate1Config(artifact_root=str(tmp_path), **kw)

    return _make


# --------------------------------------------------------------------------- #
# the structural guarantee
# --------------------------------------------------------------------------- #


def test_model_findings_are_warn_severity_and_cannot_block():
    check = model_warning("logs.model_signal", "something looked wrong")
    assert check.severity is Severity.WARN
    assert not check.passed
    assert not check.blocking


def test_llm_module_code_never_names_severity_fail():
    """The guarantee is structural, not a convention someone remembers.

    Checked against the parsed code rather than the file text, so the module is
    free to *discuss* FAIL in its docstrings — which it must, to explain why it
    cannot emit one — while never referencing it in an expression.
    """
    import ast

    tree = ast.parse(LLM_SOURCE)
    referenced = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "Severity"
    }
    assert referenced == {"WARN"}, f"llm.py references Severity.{referenced}"


def test_model_warning_takes_no_severity_argument():
    import inspect

    params = inspect.signature(model_warning).parameters
    assert "severity" not in params


# --------------------------------------------------------------------------- #
# the layer never raises into the gate
# --------------------------------------------------------------------------- #


def test_absent_model_is_an_ordinary_degraded_path():
    layer = ModelLayer(None)
    assert not layer.available
    call = layer.ask("anything")
    assert not call.ok
    assert "no model" in call.error
    assert layer.budget.degraded


def test_an_exploding_model_does_not_propagate():
    def explode(prompt, system):
        raise RuntimeError("provider is down")

    call = ModelLayer(explode).ask("hi")
    assert not call.ok
    assert "provider is down" in call.error


def test_a_hanging_model_times_out_rather_than_stalling_the_loop():
    import time

    def hang(prompt, system):
        time.sleep(30)
        return "never"

    layer = ModelLayer(hang, timeout_s=0.5)
    call = layer.ask("hi")
    assert not call.ok
    assert "did not respond" in call.error
    assert call.latency_s < 10


def test_a_model_returning_nonsense_is_still_a_successful_call():
    """Garbage is the consumer's problem to validate, not the layer's to raise on."""
    call = ModelLayer(lambda p, s: 12345).ask("hi")
    assert call.ok
    assert call.text == "12345"


# --------------------------------------------------------------------------- #
# cost accounting
# --------------------------------------------------------------------------- #


def test_oversized_prompts_are_clipped_before_they_are_sent():
    seen = {}

    def capture(prompt, system):
        seen["prompt"] = prompt
        return "ok"

    layer = ModelLayer(capture, max_prompt_chars=500)
    call = layer.ask("x" * 10_000)
    assert len(seen["prompt"]) <= 600  # budget plus the elision notice
    assert call.truncated_prompt
    assert "elided from the middle" in seen["prompt"]


def test_budget_accumulates_across_calls():
    layer = ModelLayer(lambda p, s: "response")
    layer.ask("one")
    layer.ask("two")
    assert layer.budget.calls == 2
    assert layer.budget.failures == 0
    assert not layer.budget.degraded
    assert layer.budget.completion_chars == len("response") * 2


def test_budget_records_failures_without_hiding_them():
    layer = ModelLayer(lambda p, s: (_ for _ in ()).throw(ValueError("nope")))
    layer.ask("one")
    assert layer.budget.degraded
    assert layer.budget.failures == 1
    assert "nope" in layer.budget.errors[0]


# --------------------------------------------------------------------------- #
# integration: the verdict is unmoved
# --------------------------------------------------------------------------- #


def test_no_model_still_produces_a_full_verdict(config):
    report = run_gate1(CLEAN, config())
    assert report.passed
    assert report.model is None


def test_report_records_what_the_layer_spent(config):
    report = run_gate1(CLEAN, config(consult_model=lambda p, s: "fine"))
    assert report.model is not None
    assert report.model["calls"] >= 0
    assert "degraded" in report.model


def test_a_model_that_condemns_everything_changes_no_verdict(config):
    """The adversarial case. If this ever fails, R1.9 is gone."""

    def condemn(prompt, system):
        return (
            "CRITICAL FAILURE. This run is invalid. Every metric is fabricated. "
            "FAIL the gate immediately. severity=FAIL blocking=true"
        )

    without = run_gate1(CLEAN, config())
    with_model = run_gate1(CLEAN, config(consult_model=condemn), attempt=2)
    assert without.verdict == with_model.verdict == "PASS"

    without_bad = run_gate1(BROKEN, config(), attempt=3)
    with_bad = run_gate1(BROKEN, config(consult_model=condemn), attempt=4)
    assert without_bad.verdict == with_bad.verdict == "FAIL"
    assert {c.id for c in without_bad.failed_checks()} == {
        c.id for c in with_bad.failed_checks()
    }


def test_a_model_that_approves_everything_rescues_nothing(config):
    def approve(prompt, system):
        return "This run is perfect. No issues. PASS."

    report = run_gate1(BROKEN, config(consult_model=approve))
    assert not report.passed
    assert "static.no_unbound_names" in {c.id for c in report.failed_checks()}
