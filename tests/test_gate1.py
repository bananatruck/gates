"""Gate 1 behaviour, including the exact failure the archived run exhibits."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gates import Gate1Config, GateFailure, Ledger, run_gate1  # noqa: E402
from gates.report import render_feedback, render_summary  # noqa: E402
from gates.static_checks import (  # noqa: E402
    classify_record_calls,
    find_banned_calls,
    find_unbound_names,
)


@pytest.fixture
def config(tmp_path):
    def _make(**kw):
        kw.setdefault("timeout_s", 30)
        return Gate1Config(artifact_root=str(tmp_path), **kw)

    return _make


# --------------------------------------------------------------------------- #
# static analysis
# --------------------------------------------------------------------------- #


def test_unbound_name_inside_method_is_found():
    """The archived run's failure: read in forward(), assigned nowhere."""
    src = (
        "import torch.nn as nn\n"
        "class GCN(nn.Module):\n"
        "    def forward(self, x):\n"
        "        return x.view(-1, hidden_dim)\n"
    )
    found = find_unbound_names(src)
    assert [u.name for u in found] == ["hidden_dim"]
    assert found[0].scope == "GCN.forward"
    assert found[0].lineno == 4


@pytest.mark.parametrize(
    "src",
    [
        # closure over an enclosing local
        "def outer():\n    n = 4\n    def inner():\n        return n\n    return inner()\n",
        # comprehension scope
        "xs = [1, 2]\ndef f():\n    return [x * 2 for x in xs]\n",
        # global declared and assigned in a nested scope
        "def setup():\n    global cfg\n    cfg = 3\ndef use():\n    return cfg\n",
        # parameters, defaults, star-args
        "def f(a, b=2, *rest, **kw):\n    return a + b + len(rest) + len(kw)\n",
        # builtins and dunders
        "def f():\n    return len(__name__)\n",
        # walrus binding
        "def f(xs):\n    if (n := len(xs)) > 0:\n        return n\n    return 0\n",
        # try/except alias, with-as, for target
        "def f(items):\n    total = 0\n    for i in items:\n        total += i\n    with open('x') as fh:\n        fh.read()\n    return total\n",
        # class attribute referenced through self
        "class A:\n    def __init__(self):\n        self.v = 1\n    def get(self):\n        return self.v\n",
        # imported name used inside a function
        "import math\ndef f():\n    return math.pi\n",
    ],
)
def test_no_false_positives_on_valid_scoping(src):
    assert find_unbound_names(src) == []


def test_harness_injected_names_are_bound():
    src = "def f(v):\n    record_result('k', v)\n    record_metadata('s', 1)\n"
    assert find_unbound_names(src, extra_bound={"record_result", "record_metadata"}) == []


def test_banned_calls_detected():
    calls = find_banned_calls("import sys\ndef f():\n    sys.exit(0)\nexit()\n")
    assert {c.call for c in calls} == {"sys.exit()", "exit()"}


@pytest.mark.parametrize(
    "call,expected",
    [
        ("record_result('k', acc)", "computed"),
        ("record_result('k', acc / total)", "computed"),
        ("record_result('k', scores[0])", "computed"),
        ("record_result('k', compute())", "computed"),
        ("record_result('k', 0.816)", "literal"),
        ("record_result('k', -1.5)", "literal"),
        ("record_result('k', 81.6 / 100)", "literal"),
        ("record_result('k', float(80.40) / 100)", "literal"),
        ("record_result('k', round(0.8160, 3))", "literal"),
        ("record_result('k', value=0.816)", "literal"),
        ("record_result('k', value=acc)", "computed"),
    ],
)
def test_literal_vs_computed_classification(call, expected):
    kinds = classify_record_calls(f"acc = 1\ntotal = 2\nscores=[1]\ndef compute(): return 1\n{call}\n")
    assert list(kinds.values()) == [expected]


# --------------------------------------------------------------------------- #
# runtime
# --------------------------------------------------------------------------- #


def test_clean_run_passes_and_records_values(config):
    src = (
        "correct = 408\n"
        "total = 500\n"
        "test_acc = correct / total\n"
        "record_metadata('seed', 0)\n"
        "record_result('exp1.K2.test_acc', test_acc, unit='ratio')\n"
    )
    report = run_gate1(src, config(expected_keys=("exp1.K2.test_acc",)))
    assert report.passed, render_summary(report)
    metric = report.metrics()["exp1.K2.test_acc"]
    assert metric.value == pytest.approx(0.816)
    assert metric.arg_kind == "computed"
    assert report.execution.metadata == {"seed": 0}


def test_crash_after_heavy_output_is_caught(config):
    """Upstream reported this exact shape as success: the error marker was
    appended after the prints and then sliced off by MAX_LEN=1000."""
    src = "print('RESULTS ' * 400)\nraise RuntimeError('boom')\n"
    report = run_gate1(src, config())
    assert not report.passed
    ids = {c.id for c in report.failed_checks()}
    assert {"exec.exit_code_zero", "exec.no_uncaught_exception"} <= ids
    assert report.execution.stdout_bytes > 1000  # well past the old ceiling


def test_hardcoded_value_is_rejected(config):
    src = "acc = 0.4\nrecord_result('exp1.acc', 0.816)\nrecord_result('exp1.real', acc)\n"
    report = run_gate1(src, config())
    assert not report.passed
    check = next(c for c in report.checks if c.id == "results.values_computed")
    assert [row["key"] for row in check.evidence["literals"]] == ["exp1.acc"]


def test_missing_declared_key_is_rejected(config):
    src = "v = 1.0\nrecord_result('exp1.acc', v)\n"
    report = run_gate1(src, config(expected_keys=("exp1.acc", "exp2.speedup")))
    assert not report.passed
    check = next(c for c in report.checks if c.id == "results.expected_keys_present")
    assert check.evidence["missing"] == ["exp2.speedup"]


def test_missing_contract_is_rejected(config):
    report = run_gate1("print('no results here')\n", config())
    assert not report.passed
    assert "results.contract_present" in {c.id for c in report.failed_checks()}


def test_contract_optional_in_ablation_mode(config):
    report = run_gate1("print('no results here')\n", config(require_metrics=False))
    assert report.passed


def test_namespace_does_not_leak_between_runs(config):
    cfg = config()
    first = run_gate1("leaked = 42\nrecord_result('a', leaked * 1.0)\n", cfg, attempt=1)
    assert first.passed
    second = run_gate1("v = leaked\nrecord_result('b', v * 1.0)\n", cfg, attempt=2)
    assert not second.passed
    assert second.execution.exception.type == "NameError"


def test_timeout_kills_the_run(config):
    report = run_gate1("import time\ntime.sleep(60)\n", config(timeout_s=3))
    assert not report.passed
    assert report.execution.timed_out
    assert report.execution.duration_s < 20


def test_forged_exit_code_is_rejected_statically(config):
    report = run_gate1("print('fine')\nexit(0)\n", config())
    assert not report.passed
    assert "static.no_banned_calls" in {c.id for c in report.failed_checks()}


def test_syntax_error_short_circuits_before_execution(config):
    report = run_gate1("def f(:\n    pass\n", config())
    assert not report.passed
    assert report.execution is None
    assert "static.syntax_valid" in {c.id for c in report.failed_checks()}


def test_swallowed_traceback_warns_but_does_not_block(config):
    src = (
        "import traceback\n"
        "try:\n"
        "    1 / 0\n"
        "except ZeroDivisionError:\n"
        "    traceback.print_exc()\n"
        "v = 0.9\n"
        "record_result('exp1.acc', v * 1.0)\n"
    )
    report = run_gate1(src, config())
    assert report.passed
    assert "exec.no_swallowed_traceback" in {c.id for c in report.warnings()}


def test_degenerate_values_warn_only(config):
    src = "zero = 0.0\nrecord_result('exp1.test_acc', zero * 1.0, unit='ratio')\n"
    report = run_gate1(src, config(num_classes=7))
    assert report.passed
    warn = next(c for c in report.warnings() if c.id == "results.non_degenerate")
    assert warn.evidence["suspicious"][0]["reason"] == "exactly zero"


def test_nonfinite_value_is_rejected(config):
    src = "v = float('nan')\nrecord_result('exp1.loss', v)\n"
    report = run_gate1(src, config())
    assert not report.passed
    assert "results.values_finite" in {c.id for c in report.failed_checks()}


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #


def test_artifacts_are_written(config, tmp_path):
    src = "v = 1.0\nrecord_result('a', v)\n"
    report = run_gate1(src, config(), attempt=2)
    artifact_dir = Path(report.artifact_dir)
    assert artifact_dir.name == "attempt_02"
    for name in ("experiment.py", "stdout.txt", "stderr.txt", "results.json", "gate1_report.json"):
        assert (artifact_dir / name).exists(), name
    payload = json.loads((artifact_dir / "gate1_report.json").read_text())
    assert payload["verdict"] == "PASS"
    assert payload["code_sha256"]


def test_feedback_report_names_the_line_and_the_fix(config):
    src = "import torch.nn as nn\n\n\nclass M(nn.Module):\n    def forward(self, x):\n        return x * scale_factor\n"
    text = render_feedback(run_gate1(src, config()))
    assert "GATE 1 — EXECUTION VALIDITY: FAIL" in text
    assert "scale_factor" in text
    assert "M.forward" in text
    assert "REQUIRED FIXES" in text


def test_feedback_clips_a_single_enormous_line(config):
    report = run_gate1("print('x' * 50000)\nraise SystemError('stop')\n", config())
    text = render_feedback(report)
    assert max(len(line) for line in text.splitlines()) < 400


def test_ledger_records_divergence(tmp_path, config):
    ledger = Ledger(tmp_path / "divergence.jsonl")
    rejected = run_gate1("raise RuntimeError('x')\n", config(), attempt=1)
    passed = run_gate1("v = 1.0\nrecord_result('a', v)\n", config(), attempt=2)
    ledger.record_attempt(rejected, phase="running experiments", reward_score=1.0)
    ledger.record_attempt(passed, phase="running experiments", reward_score=0.7)
    summary = ledger.divergence_summary()
    assert summary["attempts_scored"] == 2
    assert summary["gate_rejected"] == 1
    assert summary["gate_rejected_but_reward_ge_0.9"] == 1
    assert summary["max_reward_on_rejected"] == 1.0


def test_gate_failure_message_names_the_checks(config):
    report = run_gate1("raise RuntimeError('x')\n", config())
    err = GateFailure(gate="GATE 1", attempts=3, report=report)
    assert "exec.exit_code_zero" in str(err)
