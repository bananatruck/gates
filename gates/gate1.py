"""Gate 1 — execution validity.

Answers one question: *did this code actually run to completion, and were the
numbers it reports produced by this run rather than inherited, hardcoded, or
invented?*

No model is consulted. The runtime already knows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from . import static_checks
from .runner import DEFAULT_TIMEOUT_S, code_sha256, run_experiment
from .schema import (
    HARNESS_INJECTED_NAMES,
    CheckResult,
    ExecutionRecord,
    GateReport,
    MetricRecord,
    Severity,
    decide,
)

GATE_NAME = "GATE 1 — EXECUTION VALIDITY"

_TRACEBACK_MARKER = "Traceback (most recent call last)"


@dataclass
class Gate1Config:
    """Everything Gate 1 needs to know. No scaffold types appear here."""

    #: Rewrites the ML engineer gets before the gate gives up on this phase.
    max_attempts: int = 3
    timeout_s: int = DEFAULT_TIMEOUT_S
    #: Metric keys the plan declared. ``None`` means "any, but at least one".
    expected_keys: tuple[str, ...] | None = None
    #: When False, an experiment that records nothing still passes. Used only by
    #: the channel-fidelity ablation, where the point is to measure what a
    #: degraded channel produces.
    require_metrics: bool = True
    #: Enables the chance-level arm of ``results.non_degenerate``.
    num_classes: int | None = None
    cwd: str | None = None
    python: str | None = None
    artifact_root: str = "gate_artifacts"
    #: Extra names the harness is known to inject, for scaffolds that add their
    #: own runtime helpers.
    extra_bound_names: frozenset[str] = field(default_factory=frozenset)

    def attempt_dir(self, attempt: int) -> Path:
        return Path(self.artifact_root) / "gate1" / f"attempt_{attempt:02d}"


def run_gate1(source: str, config: Gate1Config, attempt: int = 1) -> GateReport:
    """Run every Gate 1 check against ``source`` and return the verdict.

    Static checks run first and short-circuit: rejecting a program with an
    unbound name should not cost a 45-minute training run.
    """
    artifact_dir = config.attempt_dir(attempt)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    checks: list[CheckResult] = []
    bound = frozenset({"record_result", "record_metadata"}) | config.extra_bound_names

    syntax = _check_syntax(source)
    checks.append(syntax)
    if syntax.passed:
        checks.append(_check_unbound_names(source, bound))
        checks.append(_check_banned_calls(source))

    report = GateReport(
        gate=GATE_NAME,
        verdict=decide(checks),
        attempt=attempt,
        max_attempts=config.max_attempts,
        checks=checks,
        artifact_dir=str(artifact_dir),
        code_sha256=code_sha256(source),
    )
    if not report.passed:
        # Rejected before execution. Nothing ran, so there is nothing to record.
        _write_report(report, artifact_dir)
        return report

    execution = run_experiment(
        source,
        artifact_dir,
        timeout_s=config.timeout_s,
        cwd=config.cwd,
        python=config.python,
    )
    _annotate_provenance(source, execution)

    checks.extend(
        [
            _check_within_budget(execution, config),
            _check_exit_code(execution),
            _check_no_uncaught_exception(execution),
            _check_no_swallowed_traceback(execution),
            _check_clean_namespace(execution),
            _check_contract_present(execution, config),
        ]
    )
    if execution.metrics:
        checks.extend(
            [
                _check_expected_keys(execution, config),
                _check_values_computed(execution),
                _check_values_finite(execution),
                _check_non_degenerate(execution, config),
            ]
        )
    checks.append(_check_untruncated(execution))
    checks.append(_record_environment(execution))

    report.checks = checks
    report.execution = execution
    report.verdict = decide(checks)
    _write_report(report, artifact_dir)
    return report


# --------------------------------------------------------------------------- #
# static checks
# --------------------------------------------------------------------------- #


def _check_syntax(source: str) -> CheckResult:
    try:
        static_checks.parse(source)
    except SyntaxError as exc:
        return CheckResult(
            id="static.syntax_valid",
            passed=False,
            severity=Severity.FAIL,
            message=f"{exc.msg} at line {exc.lineno}",
            evidence={
                "lineno": exc.lineno,
                "offset": exc.offset,
                "source_line": (exc.text or "").rstrip(),
            },
        )
    return CheckResult(
        id="static.syntax_valid", passed=True, severity=Severity.FAIL, message="compiles"
    )


def _check_unbound_names(source: str, bound: frozenset[str]) -> CheckResult:
    unbound = static_checks.find_unbound_names(source, extra_bound=bound)
    if not unbound:
        return CheckResult(
            id="static.no_unbound_names",
            passed=True,
            severity=Severity.FAIL,
            message="every referenced name resolves",
        )
    names = ", ".join(sorted({u.name for u in unbound}))
    return CheckResult(
        id="static.no_unbound_names",
        passed=False,
        severity=Severity.FAIL,
        message=f"name(s) read but never bound: {names}",
        evidence={
            "names": [
                {
                    "name": u.name,
                    "lineno": u.lineno,
                    "scope": u.scope,
                    "source_line": u.source_line,
                }
                for u in unbound
            ]
        },
    )


def _check_banned_calls(source: str) -> CheckResult:
    banned = static_checks.find_banned_calls(source)
    if not banned:
        return CheckResult(
            id="static.no_banned_calls",
            passed=True,
            severity=Severity.FAIL,
            message="no calls that forge an exit code",
        )
    return CheckResult(
        id="static.no_banned_calls",
        passed=False,
        severity=Severity.FAIL,
        message=f"{banned[0].call} would forge a clean exit code",
        evidence={
            "calls": [
                {"call": b.call, "lineno": b.lineno, "source_line": b.source_line}
                for b in banned
            ]
        },
    )


# --------------------------------------------------------------------------- #
# runtime checks
# --------------------------------------------------------------------------- #


def _check_within_budget(execution: ExecutionRecord, config: Gate1Config) -> CheckResult:
    return CheckResult(
        id="exec.completed_within_budget",
        passed=not execution.timed_out,
        severity=Severity.FAIL,
        message=(
            f"killed after {config.timeout_s}s"
            if execution.timed_out
            else f"finished in {execution.duration_s:.1f}s"
        ),
        evidence={"timeout_s": config.timeout_s, "duration_s": round(execution.duration_s, 3)},
    )


def _check_exit_code(execution: ExecutionRecord) -> CheckResult:
    ok = execution.exit_code == 0
    return CheckResult(
        id="exec.exit_code_zero",
        passed=ok,
        severity=Severity.FAIL,
        message="exited 0" if ok else f"exited {execution.exit_code}",
        evidence={"exit_code": execution.exit_code},
    )


def _check_no_uncaught_exception(execution: ExecutionRecord) -> CheckResult:
    exc = execution.exception
    if exc is None:
        return CheckResult(
            id="exec.no_uncaught_exception",
            passed=True,
            severity=Severity.FAIL,
            message="ran to completion",
        )
    return CheckResult(
        id="exec.no_uncaught_exception",
        passed=False,
        severity=Severity.FAIL,
        message=f"{exc.type}: {exc.message}",
        evidence={
            "type": exc.type,
            "message": exc.message,
            "filename": exc.filename,
            "lineno": exc.lineno,
            "function": exc.function,
            "source_line": exc.source_line,
        },
    )


def _check_no_swallowed_traceback(execution: ExecutionRecord) -> CheckResult:
    """A traceback on stderr from a run that still exited 0.

    The experiment caught its own error and carried on. Not fatal — this is the
    "errors that aren't really code-breaking" case — but the writer must not
    present the surrounding numbers as if nothing happened.
    """
    stderr = execution.stderr_text(limit=200_000)
    hit = _TRACEBACK_MARKER in stderr
    if not hit or execution.exception is not None:
        return CheckResult(
            id="exec.no_swallowed_traceback",
            passed=True,
            severity=Severity.WARN,
            message="no caught-and-ignored traceback on stderr",
        )
    lines = [ln for ln in stderr.splitlines() if ln.strip()]
    return CheckResult(
        id="exec.no_swallowed_traceback",
        passed=False,
        severity=Severity.WARN,
        message="the experiment caught an exception, printed it, and continued",
        evidence={
            "occurrences": stderr.count(_TRACEBACK_MARKER),
            "last_line": lines[-1] if lines else "",
            "stderr_path": execution.stderr_path,
        },
    )


def _check_clean_namespace(execution: ExecutionRecord) -> CheckResult:
    """The experiment started with an empty namespace, so no name it read could
    have come from a previous attempt."""
    if not execution.initial_namespace:
        return CheckResult(
            id="env.clean_namespace",
            passed=False,
            severity=Severity.FAIL,
            message="harness did not report the initial namespace",
        )
    unexpected = sorted(set(execution.initial_namespace) - HARNESS_INJECTED_NAMES)
    return CheckResult(
        id="env.clean_namespace",
        passed=not unexpected,
        severity=Severity.FAIL,
        message=(
            "started from an empty namespace"
            if not unexpected
            else f"inherited state before execution: {', '.join(unexpected)}"
        ),
        evidence={"unexpected": unexpected},
    )


# --------------------------------------------------------------------------- #
# results-contract checks
# --------------------------------------------------------------------------- #


def _check_contract_present(execution: ExecutionRecord, config: Gate1Config) -> CheckResult:
    if execution.harness_error:
        return CheckResult(
            id="results.contract_present",
            passed=False,
            severity=Severity.FAIL,
            message=execution.harness_error,
        )
    if execution.metrics:
        return CheckResult(
            id="results.contract_present",
            passed=True,
            severity=Severity.FAIL,
            message=f"{len(execution.metrics)} metric(s) recorded",
            evidence={"keys": sorted(execution.metrics)},
        )
    if not config.require_metrics:
        return CheckResult(
            id="results.contract_present",
            passed=True,
            severity=Severity.INFO,
            message="no metrics recorded (contract not required in this mode)",
        )
    reason = (
        "the run crashed before any record_result() call completed"
        if execution.exception
        else "the experiment never called record_result()"
    )
    return CheckResult(
        id="results.contract_present",
        passed=False,
        severity=Severity.FAIL,
        message=f"no results were declared — {reason}",
    )


def _check_expected_keys(execution: ExecutionRecord, config: Gate1Config) -> CheckResult:
    if not config.expected_keys:
        return CheckResult(
            id="results.expected_keys_present",
            passed=True,
            severity=Severity.FAIL,
            message="no key contract declared by the plan",
        )
    missing = sorted(set(config.expected_keys) - set(execution.metrics))
    return CheckResult(
        id="results.expected_keys_present",
        passed=not missing,
        severity=Severity.FAIL,
        message=(
            "every declared key was recorded"
            if not missing
            else f"missing declared key(s): {', '.join(missing)}"
        ),
        evidence={"missing": missing, "recorded": sorted(execution.metrics)},
    )


def _check_values_computed(execution: ExecutionRecord) -> CheckResult:
    """A recorded value that is a source literal was typed, not measured."""
    literals = [m for m in execution.metrics.values() if m.arg_kind == "literal"]
    unknown = [m for m in execution.metrics.values() if m.arg_kind == "unknown"]
    if not literals:
        message = "every recorded value is the result of a computation"
        if unknown:
            message += f" ({len(unknown)} call site(s) could not be resolved statically)"
        return CheckResult(
            id="results.values_computed",
            passed=True,
            severity=Severity.FAIL,
            message=message,
            evidence={"unresolved": [m.key for m in unknown]},
        )
    return CheckResult(
        id="results.values_computed",
        passed=False,
        severity=Severity.FAIL,
        message=(
            f"{len(literals)} value(s) were written as literals in the source "
            f"rather than computed: {', '.join(m.key for m in literals)}"
        ),
        evidence={
            "literals": [
                {"key": m.key, "value": m.value, "lineno": m.lineno, "source_line": m.source_line}
                for m in literals
            ],
            "unresolved": [m.key for m in unknown],
        },
    )


def _check_values_finite(execution: ExecutionRecord) -> CheckResult:
    bad = [m for m in execution.metrics.values() if not m.is_finite]
    return CheckResult(
        id="results.values_finite",
        passed=not bad,
        severity=Severity.FAIL,
        message=(
            "all values are finite"
            if not bad
            else f"non-finite value(s): {', '.join(f'{m.key}={m.value}' for m in bad)}"
        ),
        evidence={"keys": [m.key for m in bad]},
    )


def _check_non_degenerate(execution: ExecutionRecord, config: Gate1Config) -> CheckResult:
    """Real measurements can still be scientifically empty.

    AutoResearchClaw reports exactly this limitation: a value registry passes
    zero-valued results because the zeros are genuine. We surface them instead
    of accepting them silently — and never block on them, because a legitimate
    zero exists.
    """
    suspicious: list[dict] = []
    chance = 1.0 / config.num_classes if config.num_classes else None
    for m in execution.metrics.values():
        if isinstance(m.value, bool) or not isinstance(m.value, (int, float)):
            continue
        if not m.is_finite:
            continue
        reason = None
        if m.value == 0.0:
            reason = "exactly zero"
        elif m.value == 1.0 and _looks_like_ratio(m):
            reason = "exactly one — perfect score"
        elif chance is not None and _looks_like_ratio(m) and math.isclose(
            m.value, chance, rel_tol=1e-9, abs_tol=1e-9
        ):
            reason = f"at chance level for {config.num_classes} classes"
        if reason:
            suspicious.append({"key": m.key, "value": m.value, "reason": reason})
    return CheckResult(
        id="results.non_degenerate",
        passed=not suspicious,
        severity=Severity.WARN,
        message=(
            "no degenerate values"
            if not suspicious
            else "; ".join(f"{s['key']} is {s['reason']}" for s in suspicious)
        ),
        evidence={"suspicious": suspicious},
    )


# --------------------------------------------------------------------------- #
# provenance (recorded, never blocking)
# --------------------------------------------------------------------------- #


def _check_untruncated(execution: ExecutionRecord) -> CheckResult:
    return CheckResult(
        id="output.untruncated",
        passed=not execution.truncated,
        severity=Severity.INFO,
        message=(
            f"{execution.stdout_bytes:,} bytes of stdout and "
            f"{execution.stderr_bytes:,} bytes of stderr captured in full"
        ),
        evidence={
            "stdout_bytes": execution.stdout_bytes,
            "stderr_bytes": execution.stderr_bytes,
            "stdout_path": execution.stdout_path,
            "stderr_path": execution.stderr_path,
        },
    )


def _record_environment(execution: ExecutionRecord) -> CheckResult:
    return CheckResult(
        id="env.provenance",
        passed=True,
        severity=Severity.INFO,
        message="execution environment recorded",
        evidence=dict(execution.environment),
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _annotate_provenance(source: str, execution: ExecutionRecord) -> None:
    """Decide, from the source, whether each recorded value was computed."""
    try:
        kinds = static_checks.classify_record_calls(source)
    except SyntaxError:
        return
    for metric in execution.metrics.values():
        metric.arg_kind = kinds.get(metric.lineno or -1, "unknown")


def _looks_like_ratio(metric: MetricRecord) -> bool:
    if metric.unit in {"ratio", "accuracy", "fraction", "percent"}:
        return True
    key = metric.key.lower()
    return any(t in key for t in ("acc", "f1", "auc", "precision", "recall", "rate"))


def _write_report(report: GateReport, artifact_dir: Path) -> None:
    (artifact_dir / "gate1_report.json").write_text(report.to_json(), encoding="utf-8")
