"""Gate 1 behaviour, including the exact failure the archived run exhibits."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gates import (  # noqa: E402
    Gate1Config,
    GateFailure,
    Ledger,
    Severity,
    build_registry,
    citable_values,
    load_registry,
    resolve_trace,
    run_experiment,
    run_gate1,
)
from gates.log_checks import scan_streams  # noqa: E402
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


def test_experiment_child_does_not_inherit_provider_credentials(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "parent-only-sentinel")
    source = (
        "import os\n"
        "secret_visible = os.environ.get('DEEPSEEK_API_KEY') is not None\n"
        "override_visible = os.environ.get('OPENAI_API_KEY') is not None\n"
        "safe_visible = os.environ.get('GATE_TEST_SAFE') == 'visible'\n"
        "record_result('security.parent_secret_visible', secret_visible)\n"
        "record_result('security.override_secret_visible', override_visible)\n"
        "record_result('security.safe_override_visible', safe_visible)\n"
    )
    execution = run_experiment(
        source,
        tmp_path,
        env={"OPENAI_API_KEY": "also-parent-only", "GATE_TEST_SAFE": "visible"},
        timeout_s=30,
    )

    assert execution.exit_code == 0
    assert execution.metrics["security.parent_secret_visible"].value is False
    assert execution.metrics["security.override_secret_visible"].value is False
    assert execution.metrics["security.safe_override_visible"].value is True
    artifacts = (tmp_path / "results.json").read_text()
    assert "parent-only-sentinel" not in artifacts
    assert "also-parent-only" not in artifacts


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux /proc test")
def test_experiment_child_cannot_read_parent_proc_environment(tmp_path):
    """Scrubbing the child is insufficient if it can open its parent's env."""
    sentinel = "gate1-parent-proc-sentinel-not-a-real-secret"
    program = f'''\
from gates import run_experiment
source = """import os
try:
    parent_env = open(f'/proc/{{os.getppid()}}/environ', 'rb').read()
except OSError:
    parent_env = b''
visible = b'{sentinel}' in parent_env
record_result('security.parent_proc_visible', visible)
"""
record = run_experiment(source, {str(tmp_path)!r}, timeout_s=30)
print(record.metrics['security.parent_proc_visible'].value)
'''
    environment = os.environ.copy()
    environment["GATE_TEST_SECRET_INITIAL"] = sentinel

    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


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
        # the indirection the call-site check cannot see
        ("record_result('k', typed)", "constant"),
        ("record_result('k', typed / 100)", "constant"),
        ("record_result('k', value=typed)", "constant"),
        ("record_result('k', acc / typed)", "computed"),
    ],
)
def test_literal_vs_computed_classification(call, expected):
    preamble = (
        "acc = sum([1])\n"
        "total = sum([2])\n"
        "scores = [1]\n"
        "typed = 0.816\n"
        "def compute(): return 1\n"
    )
    kinds = classify_record_calls(f"{preamble}{call}\n")
    assert list(kinds.values()) == [expected]


@pytest.mark.parametrize(
    "binding",
    [
        "for acc in [0.816]:\n    pass",
        "acc = 0.0\nacc += 0.816",
        "def f(acc=0.816):\n    pass\nacc = 0.816",
        "import math as acc",
        "acc, _ = (0.816, 0)",
        "with open('f') as acc:\n    pass",
    ],
)
def test_a_name_bound_outside_plain_assignment_is_not_called_constant(binding):
    """The taint pass under-reports on purpose: a warning costs a rewrite."""
    kinds = classify_record_calls(f"{binding}\nrecord_result('k', acc)\n")
    assert list(kinds.values()) == ["computed"]


def test_constant_chain_terminates_on_a_self_reference():
    src = "acc = 0.816\ndef f():\n    global acc\n    acc = acc\nrecord_result('k', acc)\n"
    assert list(classify_record_calls(src).values()) == ["computed"]


# --------------------------------------------------------------------------- #
# runtime
# --------------------------------------------------------------------------- #


def test_clean_run_passes_and_records_values(config):
    src = (
        "predictions = [1] * 408 + [0] * 92\n"
        "correct = sum(predictions)\n"
        "test_acc = correct / len(predictions)\n"
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


# --------------------------------------------------------------------------- #
# log diagnostics — "log files without errors that aren't really code-breaking"
# --------------------------------------------------------------------------- #


def test_traceback_printed_to_stdout_is_caught(config):
    """A stderr-only scan misses this, and it is the common shape: the agent
    catches its own exception and prints it with the default stream."""
    src = (
        "import traceback\n"
        "try:\n"
        "    1 / 0\n"
        "except ZeroDivisionError:\n"
        "    traceback.print_exc(file=__import__('sys').stdout)\n"
        "acc = 0.5 + 0.1\n"
        "record_metadata('seed', 0)\n"
        "record_result('acc', acc)\n"
    )
    report = run_gate1(src, config())
    assert report.passed  # exited 0; this is a warning, not a failure
    check = next(c for c in report.checks if c.id == "exec.no_swallowed_traceback")
    assert not check.passed
    assert check.evidence["stdout_occurrences"] == 1
    assert check.evidence["stderr_occurrences"] == 0


def test_numerical_warning_is_surfaced(config):
    src = (
        "import sys\n"
        "print('RuntimeWarning: invalid value encountered in divide', file=sys.stderr)\n"
        "v = 1.0 * 2\n"
        "record_metadata('seed', 0)\n"
        "record_result('v', v)\n"
    )
    report = run_gate1(src, config())
    assert report.passed
    check = next(c for c in report.checks if c.id == "logs.no_error_signals")
    assert not check.passed
    assert check.evidence["counts"] == {"numerical_integrity": 1}
    # the agent has to be able to find it
    assert check.evidence["findings"][0]["stream"] == "stderr"
    assert "invalid value" in check.evidence["findings"][0]["line"]


def test_error_signals_appear_in_the_feedback_report(config):
    src = (
        "print('CUDA out of memory; falling back to CPU')\n"
        "v = 2.0 / 4\n"
        "record_metadata('seed', 1)\n"
        "record_result('v', v)\n"
    )
    report = run_gate1(src, config())
    text = render_feedback(report)
    assert "device_failure" in text
    assert "CUDA out of memory" in text


@pytest.mark.parametrize(
    "line",
    [
        "Mean Squared Error: 0.031",
        "Standard Error of the estimate: 0.02",
        "test error rate: 0.184",
        "Epoch 12 | train loss 0.31 | val acc 0.812",
        "UserWarning: TypedStorage is deprecated",
        "Converged after 40 iterations",
    ],
)
def test_ordinary_ml_logging_is_not_flagged(line):
    """False positives cost the agent a rewrite for nothing, so the patterns are
    tuned to reject ordinary experiment prose — 'Error' in a metric name above
    all."""
    assert scan_streams(line, "") == []


@pytest.mark.parametrize(
    "line,signal",
    [
        ("ValueError: could not broadcast", "printed_exception"),
        ("torch.cuda.OutOfMemoryError: CUDA out of memory", "device_failure"),
        ("ConvergenceWarning: lbfgs failed to converge", "convergence"),
        ("[ERROR] eval split was empty", "logged_error_level"),
        ("root - ERROR - could not load checkpoint", "logged_error_level"),
        ("RuntimeWarning: divide by zero encountered", "numerical_integrity"),
    ],
)
def test_real_error_signals_are_flagged(line, signal):
    findings = scan_streams(line, "")
    assert [f.signal for f in findings] == [signal]


# --------------------------------------------------------------------------- #
# repeated observations — best-epoch reported as final-epoch
# --------------------------------------------------------------------------- #


def test_repeated_record_keeps_every_observation(config):
    src = (
        "record_metadata('seed', 0)\n"
        "for i in range(4):\n"
        "    acc = 0.70 + i / 100\n"
        "    record_result('val_acc', acc)\n"
    )
    report = run_gate1(src, config())
    assert report.passed  # a warning: the gate cannot know which value is meant
    metric = report.metrics()["val_acc"]
    assert metric.call_count == 4
    assert metric.value == pytest.approx(0.73)  # the registry holds the last
    assert [o["value"] for o in metric.observations] == pytest.approx(
        [0.70, 0.71, 0.72, 0.73]
    )
    check = next(c for c in report.checks if c.id == "results.single_observation")
    assert not check.passed
    row = check.evidence["varied"][0]
    assert row["min"] == pytest.approx(0.70)
    assert row["max"] == pytest.approx(0.73)
    assert row["recorded_value"] == pytest.approx(0.73)


def test_single_record_does_not_warn(config):
    src = "record_metadata('seed', 0)\nv = 4 / 5\nrecord_result('acc', v)\n"
    report = run_gate1(src, config())
    check = next(c for c in report.checks if c.id == "results.single_observation")
    assert check.passed


def test_observation_history_is_capped(config):
    src = (
        "record_metadata('seed', 0)\n"
        "for i in range(120):\n"
        "    record_result('loss', i * 1.0)\n"
    )
    report = run_gate1(src, config())
    metric = report.metrics()["loss"]
    assert metric.call_count == 120
    assert len(metric.observations) == 50
    assert metric.observations_truncated


# --------------------------------------------------------------------------- #
# seed
# --------------------------------------------------------------------------- #


def test_missing_seed_warns_but_does_not_block(config):
    report = run_gate1("v = 4 / 5\nrecord_result('acc', v)\n", config())
    assert report.passed
    check = next(c for c in report.checks if c.id == "env.seed_recorded")
    assert not check.passed
    assert "cannot be reproduced" in check.message


def test_any_seed_key_satisfies_the_check(config):
    src = "record_metadata('numpy_seed', 7)\nv = 4 / 5\nrecord_result('acc', v)\n"
    report = run_gate1(src, config())
    check = next(c for c in report.checks if c.id == "env.seed_recorded")
    assert check.passed
    assert check.evidence["seed"] == 7


# --------------------------------------------------------------------------- #
# the limits the call-site check leaves open
# --------------------------------------------------------------------------- #


def test_value_laundered_through_a_variable_warns_but_does_not_block(config):
    """`acc = 0.816; record_result(k, acc)` is the literal check's blind spot."""
    src = (
        "record_metadata('seed', 0)\n"
        "test_acc = 0.816\n"
        "record_result('exp1.K2.test_acc', test_acc, unit='ratio')\n"
    )
    report = run_gate1(src, config(expected_keys=("exp1.K2.test_acc",)))

    assert report.passed, render_summary(report)
    computed = next(c for c in report.checks if c.id == "results.values_computed")
    assert computed.passed
    traced = next(c for c in report.checks if c.id == "results.values_traced")
    assert not traced.passed
    assert traced.severity is Severity.WARN
    assert traced.evidence["constant_derived"][0]["key"] == "exp1.K2.test_acc"
    assert report.metrics()["exp1.K2.test_acc"].arg_kind == "constant"


def test_a_real_measurement_is_not_called_constant(config):
    src = (
        "record_metadata('seed', 0)\n"
        "predictions = [1] * 408 + [0] * 92\n"
        "record_result('exp1.K2.test_acc', sum(predictions) / len(predictions))\n"
    )
    report = run_gate1(src, config())
    traced = next(c for c in report.checks if c.id == "results.values_traced")
    assert traced.passed, traced.message


def test_keys_the_plan_never_declared_are_reported(config):
    src = (
        "record_metadata('seed', 0)\n"
        "record_result('exp1.acc', sum([1]) / 2)\n"
        "record_result('leftover.from_earlier_phase', sum([3]) / 2)\n"
    )
    report = run_gate1(src, config(expected_keys=("exp1.acc",)))

    assert report.passed, render_summary(report)
    declared = next(c for c in report.checks if c.id == "results.declared_keys_only")
    assert not declared.passed
    assert declared.severity is Severity.WARN
    assert declared.evidence["undeclared"] == ["leftover.from_earlier_phase"]


def test_exactly_the_declared_keys_raises_nothing(config):
    src = "record_metadata('seed', 0)\nrecord_result('exp1.acc', sum([1]) / 2)\n"
    report = run_gate1(src, config(expected_keys=("exp1.acc",)))
    declared = next(c for c in report.checks if c.id == "results.declared_keys_only")
    assert declared.passed


# --------------------------------------------------------------------------- #
# registry — "typed and hashed to the run that produced it"
# --------------------------------------------------------------------------- #


def test_a_statically_rejected_run_still_gets_a_registry(config):
    """Nothing ran, so the registry is empty — but it exists and says so.

    A consumer that reads registry.json and forgets to read the verdict must
    find a file marked not citable rather than no file at all.
    """
    report = run_gate1("test_acc = (0.816\n", config())
    assert not report.passed
    assert not any(c.id.startswith("exec.") for c in report.checks)
    assert report.execution is None

    registry = load_registry(Path(report.artifact_dir) / "registry.json")
    assert registry["citable"] is False
    assert registry["values"] == {}
    assert registry["run"]["code_sha256"] == report.code_sha256


def test_registry_binds_each_value_to_the_run(config, tmp_path):
    src = (
        "record_metadata('seed', 0)\n"
        "predictions = [1] * 408 + [0] * 92\n"
        "acc = sum(predictions) / len(predictions)\n"
        "record_result('exp1.K2.test_acc', acc, unit='ratio')\n"
    )
    report = run_gate1(src, config(task_ref="reproduce SGC on Cora"), attempt=1)
    assert report.passed

    registry = load_registry(Path(report.artifact_dir) / "registry.json")
    assert registry["citable"] is True
    assert registry["run"]["code_sha256_verified"] is True
    assert registry["run"]["seed"] == 0
    assert registry["run"]["argv"]

    entry = registry["values"]["exp1.K2.test_acc"]
    assert entry["value"] == pytest.approx(0.816)
    assert entry["unit"] == "ratio"
    assert entry["type"] == "float"
    assert entry["provenance"]["arg_kind"] == "computed"
    assert entry["provenance"]["run_id"] == registry["run"]["run_id"]

    # every link of the chain resolves, so this value is causally traced rather
    # than merely present in a log
    assert entry["chain_complete"] is True
    assert [link["link"] for link in entry["chain"]] == ["task", "command", "log", "value"]
    assert registry["chain_integrity"]["rate"] == 1.0

    found = resolve_trace(registry, entry["trace_id"])
    assert found["key"] == "exp1.K2.test_acc"


def test_trace_ids_differ_between_runs_of_identical_code(config):
    """Same source, two executions. A value from run A must not be presentable as
    a value from run B, which is what makes a backfilled number detectable."""
    cfg = config()
    src = "record_metadata('seed', 0)\nv = 4 / 5\nrecord_result('acc', v)\n"
    first = run_gate1(src, cfg, attempt=1)
    second = run_gate1(src, cfg, attempt=2)
    assert first.code_sha256 == second.code_sha256  # identical source
    assert first.execution.run_id != second.execution.run_id
    assert first.metrics()["acc"].trace_id != second.metrics()["acc"].trace_id


def test_rejected_run_yields_no_citable_values(config):
    report = run_gate1("acc = 0.4\nrecord_result('acc', 0.816)\n", config())
    assert not report.passed
    registry = build_registry(report)
    assert registry["citable"] is False
    # the number is on record, with its provenance, but nothing may cite it
    assert registry["values"]["acc"]["provenance"]["arg_kind"] == "literal"
    assert citable_values(registry) == {}


def test_chain_reports_the_missing_link_when_no_task_is_supplied(config):
    src = "record_metadata('seed', 0)\nv = 4 / 5\nrecord_result('acc', v)\n"
    report = run_gate1(src, config())  # no task_ref
    registry = build_registry(report)
    entry = registry["values"]["acc"]
    assert entry["chain_complete"] is False
    assert registry["chain_integrity"]["missing_links"] == {"acc": ["task"]}
    task_link = entry["chain"][0]
    assert task_link["resolved"] is False
    assert "no task reference" in task_link["why"]


def test_executed_source_is_hashed_by_the_process_that_ran_it(config):
    src = "record_metadata('seed', 0)\nv = 4 / 5\nrecord_result('acc', v)\n"
    report = run_gate1(src, config())
    check = next(c for c in report.checks if c.id == "env.code_identity")
    assert check.passed
    assert report.execution.code_sha256 == report.code_sha256


def test_timeout_message_does_not_blame_the_contract(config):
    """A killed run never gets to write results.json. Saying it 'never called
    record_result' sends the agent to fix the wrong thing."""
    report = run_gate1("import time\ntime.sleep(60)\n", config(timeout_s=3))
    assert not report.passed
    check = next(c for c in report.checks if c.id == "results.contract_present")
    assert "killed at the timeout" in check.message


# --------------------------------------------------------------------------- #
# the recording API must not be shadowed — found live
# --------------------------------------------------------------------------- #

SHADOWED = (
    "import numpy as np\n"
    "\n"
    "def record_result(key, value, unit=None):\n"
    "    print(f'{key}: {value}')\n"
    "\n"
    "acc = 0.33\n"
    "record_result('exp1.test_acc', acc)\n"
)


def test_redefining_the_recording_api_is_caught_statically(config):
    """Observed live: a code model wrote its own record_result that printed,
    called it four times, exited 0, and recorded nothing.

    The run was rejected on results.contract_present -- "the experiment never
    called record_result()" -- which is true of the harness's function and tells
    the agent the wrong thing about its own code. Caught before execution now.
    """
    report = run_gate1(SHADOWED, config())
    assert not report.passed
    assert report.execution is None, "must be rejected before it costs a run"
    check = next(
        c for c in report.checks if c.id == "results.contract_not_shadowed"
    )
    assert not check.passed
    assert check.evidence["shadowed"][0]["name"] == "record_result"
    assert check.evidence["shadowed"][0]["kind"] == "function"
    assert check.evidence["shadowed"][0]["lineno"] == 3


def test_the_feedback_names_the_real_mistake(config):
    text = render_feedback(run_gate1(SHADOWED, config()))
    assert "record_result" in text
    assert "Delete your own definition" in text
    assert "redefined as a function at line 3" in text


@pytest.mark.parametrize(
    "src",
    [
        "record_result = print\nrecord_result('a', 1)\n",
        "from mymod import record_metadata\n",
        "import json as record_result\n",
    ],
)
def test_assignment_and_import_shadowing_are_caught_too(src, config):
    report = run_gate1(src, config())
    assert "results.contract_not_shadowed" in {
        c.id for c in report.failed_checks()
    }


def test_ordinary_use_of_the_api_is_not_flagged(config):
    src = (
        "record_metadata('seed', 0)\n"
        "v = 408 / 500\n"
        "record_result('exp1.acc', v, unit='ratio')\n"
    )
    report = run_gate1(src, config())
    assert report.passed, render_summary(report)


def test_a_local_variable_named_similarly_is_not_flagged(config):
    """The check looks for the injected names, not anything resembling them."""
    src = (
        "record_results_later = True\n"
        "v = 0.5\n"
        "record_metadata('seed', 1)\n"
        "record_result('exp1.acc', v * 2)\n"
    )
    assert run_gate1(src, config()).passed
