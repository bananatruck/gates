"""Gate 1 — execution validity.

Answers one question: *did this code actually run to completion, and were the
numbers it reports produced by this run rather than inherited, hardcoded, or
invented?*

No model is consulted. The runtime already knows.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import llm_report, llm_scan, log_checks, static_checks
from .llm import (
    DEFAULT_MAX_PROMPT_CHARS,
    DEFAULT_TIMEOUT_S as DEFAULT_MODEL_TIMEOUT_S,
    ModelFn,
    ModelLayer,
)
from .errors import GateError
from .registry import write_registry
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

#: Streams are read up to this many characters for log diagnostics. A run that
#: prints more than this has its head scanned; the full capture stays on disk.
_LOG_SCAN_CHARS = 2_000_000


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
    #: The task or plan text this experiment implements. Hashed into the registry
    #: as the first link of the provenance chain; without it that link is
    #: reported unresolved rather than assumed.
    task_ref: str | None = None
    #: When False, an experiment that declares no seed still passes silently.
    #: Left on by default: a run with no seed cannot be re-executed to check its
    #: own numbers.
    require_seed: bool = True
    #: The LLM layer's model. Supplied by the host adapter — ``gates`` imports no
    #: client. Absent, the gate still issues a full verdict and falls back to the
    #: deterministic feedback template, saying that it did. See ``llm.py`` for
    #: why a required model layer does not put a model in the verdict.
    consult_model: ModelFn | None = None
    model_timeout_s: float = DEFAULT_MODEL_TIMEOUT_S
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS
    #: The settings the run was configured with, fixed before it executed
    #: (the 09-29 review's Q4). When given, every ``record_setting`` key must
    #: be declared here with the value the run records, so a setting cannot be
    #: a number the run produced. ``None`` means no config was fixed, and
    #: ``results.settings_declared`` is absent rather than green.
    declared_settings: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.declared_settings is None:
            return
        for key, value in self.declared_settings.items():
            if not isinstance(key, str) or not isinstance(value, (bool, int, float, str)):
                raise GateError(
                    f"declared setting {key!r} must map a string key to a number, "
                    f"string or bool, not {type(value).__name__}"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise GateError(f"declared setting {key!r} is {value}, not a finite number")
        # Copied, so the config the run is checked against is the one it was
        # given, whatever the caller does with its own dict later. A plain dict
        # rather than a read-only view, because the host pickles its state.
        self.declared_settings = dict(self.declared_settings)

    def attempt_dir(self, attempt: int) -> Path:
        return Path(self.artifact_root) / "gate1" / f"attempt_{attempt:02d}"


def run_gate1(
    source: str, config: Gate1Config, attempt: int = 1, *, rewrite: int = 0
) -> GateReport:
    """Run every Gate 1 check against ``source`` and return the verdict.

    ``rewrite`` is the agent turn this execution belongs to. It is set here,
    not by the caller afterwards, because the report and registry are written
    to disk before this returns.

    Static checks run first and short-circuit: rejecting a program with an
    unbound name should not cost a 45-minute training run.
    """
    artifact_dir = config.attempt_dir(attempt)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # The LLM layer is constructed for every run, model or not: callers get the
    # same object either way, and an absent model is an ordinary degraded path
    # rather than a branch every call site has to remember.
    model = ModelLayer(
        config.consult_model,
        timeout_s=config.model_timeout_s,
        max_prompt_chars=config.max_prompt_chars,
    )

    checks: list[CheckResult] = []
    bound = (
        frozenset({"record_result", "record_setting", "record_metadata"})
        | config.extra_bound_names
    )

    syntax = _check_syntax(source)
    checks.append(syntax)
    if syntax.passed:
        checks.append(_check_unbound_names(source, bound))
        checks.append(_check_banned_calls(source))
        checks.append(_check_contract_not_shadowed(source))

    report = GateReport(
        gate=GATE_NAME,
        verdict=decide(checks),
        attempt=attempt,
        rewrite=rewrite,
        max_attempts=config.max_attempts,
        checks=checks,
        artifact_dir=str(artifact_dir),
        code_sha256=code_sha256(source),
    )
    if not report.passed:
        # Rejected before execution. Nothing ran, so there is nothing to record
        # — but the engineer still needs a report, and this is the cheapest
        # rejection there is, so it is the one worth writing well.
        _attach_generated_fixes(report, model, source)
        _record_model_budget(report, model)
        _write_report(report, artifact_dir)
        # An empty registry, marked not citable. Written here as well as on the
        # runtime path so that "a rejected run still has a registry saying so"
        # holds for every rejection: a consumer that reads registry.json and
        # forgets to read the verdict must not find the file missing and treat
        # its absence as an unrelated error.
        write_registry(report, artifact_dir, task_ref=config.task_ref)
        return report

    execution = run_experiment(
        source,
        artifact_dir,
        timeout_s=config.timeout_s,
        cwd=config.cwd,
        python=config.python,
    )
    _annotate_provenance(source, execution)

    stdout = execution.stdout_text(limit=_LOG_SCAN_CHARS)
    stderr = execution.stderr_text(limit=_LOG_SCAN_CHARS)

    checks.extend(
        [
            _check_within_budget(execution, config),
            _check_exit_code(execution),
            _check_no_uncaught_exception(execution),
            _check_code_identity(execution, report.code_sha256),
            _check_no_swallowed_traceback(execution, stdout, stderr),
            _check_log_error_signals(stdout, stderr),
        ]
    )

    # The model reads what the patterns did not flag. WARN-severity by
    # construction, and appended before decide() only because ordering the
    # report by tier reads better than ordering it by author — decide() is
    # blind to everything here either way.
    scan_check = llm_scan.build_check(
        llm_scan.scan_with_model(
            model,
            stdout,
            stderr,
            already_flagged=log_checks.scan_streams(stdout, stderr),
        )
    )
    if scan_check is not None:
        checks.append(scan_check)

    checks.extend(
        [
            _check_clean_namespace(execution),
            _check_seed_recorded(execution, config),
            _check_contract_present(execution, config),
        ]
    )
    if execution.metrics:
        checks.extend(
            [
                _check_expected_keys(execution, config),
                _check_undeclared_keys(execution, config),
                _check_values_computed(execution),
                _check_values_traced(execution),
                _check_values_finite(execution),
                _check_single_observation(execution),
                _check_no_selection(execution, source),
                _check_non_degenerate(execution, config),
            ]
        )
    if execution.settings:
        checks.append(_check_setting_single_value(execution))
        if config.declared_settings is not None:
            checks.append(_check_settings_declared(execution, config.declared_settings))
    checks.append(_check_untruncated(execution))
    guard = _check_parent_guard(execution)
    if guard is not None:
        checks.append(guard)
    checks.append(_record_environment(execution))

    report.checks = checks
    report.execution = execution
    # Verdict is fixed here, from deterministic checks alone. Anything the model
    # contributed above is WARN-severity by construction and cannot reach this.
    report.verdict = decide(checks)
    # Only now, with the verdict already fixed, does the model get asked to
    # write anything the engineer will read.
    _attach_generated_fixes(report, model, source)
    _record_model_budget(report, model)
    _write_report(report, artifact_dir)
    # Written for passing and failing runs alike; the file records which it was,
    # and a rejected registry reports itself as not citable.
    write_registry(report, artifact_dir, task_ref=config.task_ref)
    return report


# --------------------------------------------------------------------------- #
# the model layer's contributions, both after the verdict
# --------------------------------------------------------------------------- #


def _attach_generated_fixes(report: GateReport, model: ModelLayer, source: str) -> None:
    """Have the model write REQUIRED FIXES, and keep it only if it is grounded.

    Called after ``verdict`` is already set, on both return paths. A passing run
    has no fixes to write; a rejected one gets the template unless the model
    produced something every name and line number of which the report supports.
    """
    if report.passed or not model.available:
        return
    # Rejected whole rather than repaired. A fix naming a variable the code does
    # not contain sends the engineer chasing something that does not exist,
    # which is the failure this gate exists to prevent.
    llm_report.attach_fixes(
        report, llm_report.generate_fixes(model, report, source), subject="this run"
    )


def _record_model_budget(report: GateReport, model: ModelLayer) -> None:
    if model.available:
        report.model = model.budget.to_dict()


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


def _check_contract_not_shadowed(source: str) -> CheckResult:
    """The experiment must use the harness's recording API, not its own.

    Found live: an agent defined its own ``record_result`` that printed, called
    it four times, and exited 0. Nothing was recorded, and the run was rejected
    for "never calling record_result()" -- which was true of the harness's
    function and told the agent the wrong thing about its own code. Static, so
    it costs no execution, and specific, so the feedback names the real mistake.
    """
    shadowed = static_checks.find_shadowed_harness_names(source)
    if not shadowed:
        return CheckResult(
            id="results.contract_not_shadowed",
            passed=True,
            severity=Severity.FAIL,
            message="the harness recording API is not redefined",
        )
    names = ", ".join(sorted({s.name for s in shadowed}))
    return CheckResult(
        id="results.contract_not_shadowed",
        passed=False,
        severity=Severity.FAIL,
        message=(
            f"the experiment defines its own {names} — values passed to it go "
            f"nowhere the gate can see. {names} is already provided; remove the "
            f"definition and call it directly"
        ),
        evidence={
            "shadowed": [
                {"name": s.name, "lineno": s.lineno, "kind": s.kind,
                 "source_line": s.source_line}
                for s in shadowed
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


def _check_code_identity(execution: ExecutionRecord, expected_sha: str | None) -> CheckResult:
    """The source that ran is the source we hashed.

    Without this, "hashed to the run that produced it" is an assumption. With it,
    a value's attribution to a code version is checkable by a third party holding
    only the artifacts.
    """
    actual = execution.code_sha256
    if actual is None:
        return CheckResult(
            id="env.code_identity",
            passed=False,
            severity=Severity.FAIL,
            message="the harness did not report a hash of the source it executed",
            evidence={"expected": expected_sha},
        )
    ok = actual == expected_sha
    return CheckResult(
        id="env.code_identity",
        passed=ok,
        severity=Severity.FAIL,
        message=(
            f"executed source matches the recorded hash ({actual[:12]})"
            if ok
            else "the source that ran does not match the source that was submitted"
        ),
        evidence={"expected": expected_sha, "as_executed": actual},
    )


def _check_no_swallowed_traceback(
    execution: ExecutionRecord, stdout: str, stderr: str
) -> CheckResult:
    """A traceback in the logs from a run that still exited 0.

    The experiment caught its own error and carried on. Not fatal — this is the
    "errors that aren't really code-breaking" case — but the writer must not
    present the surrounding numbers as if nothing happened.

    Both streams are scanned. ``except Exception as e: print(e)`` and
    ``traceback.print_exc(file=sys.stdout)`` are the common shapes, and both put
    the evidence on stdout where a stderr-only check never sees it.
    """
    out_hits, err_hits = log_checks.count_tracebacks(stdout, stderr)
    total = out_hits + err_hits
    if not total or execution.exception is not None:
        return CheckResult(
            id="exec.no_swallowed_traceback",
            passed=True,
            severity=Severity.WARN,
            message="no caught-and-ignored traceback in the logs",
        )
    streams = ", ".join(
        s for s, n in (("stdout", out_hits), ("stderr", err_hits)) if n
    )
    return CheckResult(
        id="exec.no_swallowed_traceback",
        passed=False,
        severity=Severity.WARN,
        message=(
            f"the experiment caught {total} exception(s), printed the traceback "
            f"to {streams}, and continued"
        ),
        evidence={
            "stdout_occurrences": out_hits,
            "stderr_occurrences": err_hits,
            "stdout_path": execution.stdout_path,
            "stderr_path": execution.stderr_path,
        },
    )


def _check_log_error_signals(stdout: str, stderr: str) -> CheckResult:
    """Trouble the run reported in its own output and then carried on past.

    These do not break the process, so no exit code records them — but a metric
    computed after ``invalid value encountered`` or after a CUDA fallback does
    not mean what it appears to mean.
    """
    findings = log_checks.scan_streams(stdout, stderr)
    if not findings:
        return CheckResult(
            id="logs.no_error_signals",
            passed=True,
            severity=Severity.WARN,
            message="no error, warning or device-failure signals in the logs",
        )
    by_signal: dict[str, int] = {}
    for f in findings:
        by_signal[f.signal] = by_signal.get(f.signal, 0) + 1
    summary = ", ".join(f"{name} ×{n}" for name, n in sorted(by_signal.items()))
    return CheckResult(
        id="logs.no_error_signals",
        passed=False,
        severity=Severity.WARN,
        message=f"the logs report trouble the run continued past: {summary}",
        evidence={
            "counts": by_signal,
            "findings": [
                {
                    "signal": f.signal,
                    "note": f.note,
                    "stream": f.stream,
                    "lineno": f.lineno,
                    "line": f.line,
                }
                for f in findings
            ],
        },
    )


def _check_seed_recorded(execution: ExecutionRecord, config: Gate1Config) -> CheckResult:
    """A run with no declared seed cannot be re-executed to check its own numbers.

    Warn, never block: an experiment can be legitimately deterministic. But
    execution-grounded verification — re-running the code to confirm the claim —
    needs the seed, so its absence is a limitation the report has to carry.
    """
    seed = execution.seed()
    if seed is not None:
        return CheckResult(
            id="env.seed_recorded",
            passed=True,
            severity=Severity.WARN,
            message=f"seed recorded ({seed})",
            evidence={"seed": seed},
        )
    if not config.require_seed:
        return CheckResult(
            id="env.seed_recorded",
            passed=True,
            severity=Severity.INFO,
            message="no seed recorded (not required in this mode)",
        )
    return CheckResult(
        id="env.seed_recorded",
        passed=False,
        severity=Severity.WARN,
        message=(
            "no seed was declared, so this run cannot be reproduced and its "
            "numbers cannot be checked by re-execution"
        ),
        evidence={"metadata_keys": sorted(execution.metadata)},
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
    if execution.timed_out:
        reason = "the run was killed at the timeout before its results were written"
    elif execution.exception:
        reason = "the run crashed before any record_result() call completed"
    else:
        reason = "the experiment never called record_result()"
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


def _check_undeclared_keys(execution: ExecutionRecord, config: Gate1Config) -> CheckResult:
    """Keys the run recorded that the plan never asked for.

    ``results.expected_keys_present`` tests presence, not equality, so an
    experiment satisfies its contract while recording anything else it likes.
    That matters for a reason beyond tidiness: when a scaffold prepends an
    earlier phase's code to this one, the earlier phase's keys arrive here and
    can satisfy a contract this run never met. Surfacing them is what makes that
    blending visible in the report instead of invisible in the registry.

    A warning, not a failure. Extra measurements are ordinary and often useful;
    turning them into a rejection would cost the agent a rewrite for doing more
    work than it was asked to.
    """
    if not config.expected_keys:
        return CheckResult(
            id="results.declared_keys_only",
            passed=True,
            severity=Severity.WARN,
            message="no key contract declared by the plan",
        )
    extra = sorted(set(execution.metrics) - set(config.expected_keys))
    return CheckResult(
        id="results.declared_keys_only",
        passed=not extra,
        severity=Severity.WARN,
        message=(
            "the run recorded exactly the declared keys"
            if not extra
            else (
                f"{len(extra)} key(s) recorded that the plan did not declare: "
                f"{', '.join(extra)}"
            )
        ),
        evidence={"undeclared": extra, "declared": sorted(config.expected_keys)},
    )


def _check_values_computed(execution: ExecutionRecord) -> CheckResult:
    """A recorded value that is a source literal was typed, not measured."""
    literals = [m for m in execution.metrics.values() if m.arg_kind == "literal"]
    unknown = [m for m in execution.metrics.values() if m.arg_kind == "unknown"]
    if not literals:
        message = "no recorded value is a literal at its call site"
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


def _check_values_traced(execution: ExecutionRecord) -> CheckResult:
    """A value whose every input is a literal bound elsewhere in the source.

    ``results.values_computed`` reads the call site and nothing else, so
    ``record_result("k", 0.816)`` fails it and ``acc = 0.816`` followed by
    ``record_result("k", acc)`` does not. The only difference between those two
    programs is one indirection. This check follows the names back through their
    assignments and reports the values that bottom out in constants.

    It fails (D75). Until ``record_setting`` existed a configured batch size or
    a sweep's lambda had nowhere to go but ``record_result``, so a constant there
    was ambiguous and only warned. Now a configured value has its own call, and
    a constant recorded as a result is the fabrication this gate exists to stop.
    """
    derived = [m for m in execution.metrics.values() if m.arg_kind == "constant"]
    if not derived:
        return CheckResult(
            id="results.values_traced",
            passed=True,
            severity=Severity.FAIL,
            message="every recorded value traces back to something the run computed",
        )
    return CheckResult(
        id="results.values_traced",
        passed=False,
        severity=Severity.FAIL,
        message=(
            f"{len(derived)} value(s) resolve to source constants rather than to "
            f"anything this run measured: {', '.join(m.key for m in derived)}. "
            f"A result must come from the computation; a value the run was "
            f"configured with is recorded with record_setting(), not record_result()"
        ),
        evidence={
            "constant_derived": [
                {
                    "key": m.key,
                    "value": m.value,
                    "lineno": m.lineno,
                    "source_line": m.source_line,
                }
                for m in derived
            ]
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


def _check_single_observation(execution: ExecutionRecord) -> CheckResult:
    """A key recorded repeatedly with a changing value.

    The registry keeps the last call, but a key written once per epoch has as
    many candidate values as it has epochs — and reporting the best one as though
    it were the final one is a documented failure mode of these agents. The gate
    cannot know which value the paper will use, so it records the spread and
    requires the report to say which it means.
    """
    varied = [m for m in execution.metrics.values() if m.varied]
    if not varied:
        return CheckResult(
            id="results.single_observation",
            passed=True,
            severity=Severity.WARN,
            message="every recorded key has one unambiguous value",
        )
    rows = []
    for m in varied:
        numeric = m.numeric_observations()
        rows.append(
            {
                "key": m.key,
                "call_count": m.call_count,
                "recorded_value": m.value,
                "first": m.observations[0].get("value") if m.observations else None,
                "min": min(numeric) if numeric else None,
                "max": max(numeric) if numeric else None,
                "truncated": m.observations_truncated,
            }
        )
    return CheckResult(
        id="results.single_observation",
        passed=False,
        severity=Severity.WARN,
        message=(
            "recorded more than once with a changing value: "
            + "; ".join(f"{r['key']} ({r['call_count']} calls)" for r in rows)
            + " — the registry holds the last call, not the best one"
        ),
        evidence={"varied": rows},
    )


def _check_no_selection(execution: ExecutionRecord, source: str) -> CheckResult:
    """A recorded result picked from several, or drawn at random (S5, S7-S9).

    The best epoch, the best seed, the mean after dropping the worst seed: each
    records a number real computation produced, so no check on the value can
    object (the 09-29 review's issue 1). The source still shows the pick.

    A warning, never a failure: the maximum over validation scores that chose
    a model is honest and looks the same as the maximum over test scores. It
    reaches the writer as something the paper must state.
    """
    try:
        selected = static_checks.find_selected_record_values(source)
    except SyntaxError:
        selected = {}
    rows = [
        {"key": m.key, "lineno": m.lineno, "reason": selected[m.lineno]}
        for m in execution.metrics.values()
        if m.lineno in selected
    ]
    return CheckResult(
        id="results.no_selection",
        passed=not rows,
        severity=Severity.WARN,
        message=(
            "no recorded result is picked from several values or drawn at random"
            if not rows
            else (
                f"{len(rows)} recorded result(s) picked from several values or "
                f"drawn at random: "
                + "; ".join(f"{r['key']} ({r['reason']})" for r in rows)
            )
        ),
        evidence={"selected": rows},
    )


def _check_setting_single_value(execution: ExecutionRecord) -> CheckResult:
    """A setting key recorded with more than one value (B9).

    The registry holds one value per setting key, and ``\\setting{key}`` renders
    it. A sweep that records ``exp1.lam`` once per row leaves the last row's
    lambda there, which Gate 3 would then print beside every other row's result
    and Gate 2 would check the plan against.

    It fails where ``results.single_observation`` only warns, and the
    difference is what the last value means. A result recorded once per epoch
    is a trajectory whose last value is the final one; a setting recorded once
    per row has no final value, since each is right only for its own row. And
    the fix belongs here: the engineer can record each row under its own key,
    while the writer downstream can only leave the number out.
    """
    varied = [s for s in execution.settings.values() if s.varied]
    if not varied:
        return CheckResult(
            id="results.setting_single_value",
            passed=True,
            severity=Severity.FAIL,
            message="every setting key holds one value",
        )
    rows = [
        {
            "key": s.key,
            "call_count": s.call_count,
            "values": s.distinct_values(),
            "truncated": s.observations_truncated,
        }
        for s in varied
    ]
    return CheckResult(
        id="results.setting_single_value",
        passed=False,
        severity=Severity.FAIL,
        message=(
            f"{len(rows)} setting key(s) recorded with more than one value, so "
            f"the registry would hold only the last: "
            + "; ".join(
                f"{r['key']} ({', '.join(map(str, r['values']))})" for r in rows
            )
        ),
        evidence={"varied": rows},
    )


def declared_settings_sha256(declared: Mapping[str, Any]) -> str:
    """The hash a run's evidence records for the config it was checked against."""
    text = json.dumps(dict(declared), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _same_setting(recorded: Any, declared: Any) -> bool:
    """Numbers agree by value, a bool only with a bool, a string only exactly."""
    if isinstance(recorded, bool) or isinstance(declared, bool):
        return type(recorded) is type(declared) and recorded == declared
    if isinstance(recorded, (int, float)) and isinstance(declared, (int, float)):
        return float(recorded) == float(declared)
    return type(recorded) is type(declared) and recorded == declared


def _check_settings_declared(
    execution: ExecutionRecord, declared: Mapping[str, Any]
) -> CheckResult:
    """Every recorded setting was declared, with that value, before the run (Q4).

    ``record_setting`` is the agent's word that a number configured the run
    (D75), and Gate 3 renders it as configuration. Without a config fixed
    beforehand, a measured or invented result could be recorded as a setting
    and cited as one (red team S12). With one, a setting can only be a value
    that existed before the run did.

    It fails rather than warns because the fix is the agent's own: record the
    declared value, or leave an undeclared number out of the registry. It does
    not fail a declared key the run never records; whether the run honoured
    each declared setting is Gate 2 tier B's question.
    """
    undeclared, mismatched = [], []
    for key, setting in sorted(execution.settings.items()):
        if key not in declared:
            undeclared.append({"key": key, "recorded": setting.value})
        elif not _same_setting(setting.value, declared[key]):
            mismatched.append(
                {"key": key, "recorded": setting.value, "declared": declared[key]}
            )
    bad = [row["key"] for row in undeclared + mismatched]
    return CheckResult(
        id="results.settings_declared",
        passed=not bad,
        severity=Severity.FAIL,
        message=(
            f"every recorded setting matches the {len(declared)} declared before the run"
            if not bad
            else (
                f"{len(bad)} setting(s) differ from the config fixed before the "
                f"run: {', '.join(bad)}"
            )
        ),
        evidence={
            "undeclared": undeclared,
            "mismatched": mismatched,
            "declared_sha256": declared_settings_sha256(declared),
        },
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


_GUARD_GAPS = {
    "bypassable": "the experiment ran with CAP_SYS_PTRACE (root), which ignores the non-dumpable flag",
    "failed": "prctl refused to mark the parent non-dumpable",
    "unsupported": "this platform has no non-dumpable flag; only environment scrubbing applied",
}


def _check_parent_guard(execution: ExecutionRecord) -> CheckResult | None:
    """Whether the experiment could read this process's memory or environment (B3).

    INFO, never a verdict: the code under test did nothing wrong, the host
    running it did. The row exists so a report never implies an isolation it
    did not have. A record nothing measured emits no row.
    """
    state = execution.parent_guard
    if state is None:
        return None
    return CheckResult(
        id="env.parent_proc_guard",
        passed=state == "active",
        severity=Severity.INFO,
        message=(
            "the experiment could not read the parent process"
            if state == "active"
            else f"the experiment could read the parent process: {_GUARD_GAPS.get(state, state)}"
        ),
        evidence={"state": state},
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
    for api, store in (
        ("record_result", execution.metrics),
        ("record_setting", execution.settings),
    ):
        try:
            kinds = static_checks.classify_record_calls(source, func_name=api)
            unused = static_checks.find_unused_record_values(source, func_name=api)
        except SyntaxError:
            return
        for metric in store.values():
            metric.arg_kind = kinds.get(metric.lineno or -1, "unknown")
            if metric.arg_kind in ("constant", "computed"):
                metric.used_by_run = metric.lineno not in unused


def _looks_like_ratio(metric: MetricRecord) -> bool:
    if metric.unit in {"ratio", "accuracy", "fraction", "percent"}:
        return True
    key = metric.key.lower()
    return any(t in key for t in ("acc", "f1", "auc", "precision", "recall", "rate"))


def _write_report(report: GateReport, artifact_dir: Path) -> None:
    (artifact_dir / "gate1_report.json").write_text(report.to_json(), encoding="utf-8")
