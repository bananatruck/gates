"""Rendering of the small feedback report a gate hands back to the agent.

Deliberately narrow. It carries facts the runtime established and directives
that follow from them — no score, no praise, no model opinion. Anything the
agent cannot act on is noise that costs context.
"""

from __future__ import annotations

from .schema import CheckResult, GateReport

_MAX_EVIDENCE_ROWS = 5
_MAX_TAIL_LINES = 12
_MAX_TAIL_LINE_CHARS = 200


def render_feedback(report: GateReport, *, include_stdout_tail: bool = True) -> str:
    """The agent-facing report. Kept under roughly 40 lines on purpose."""
    # The budget counts agent rewrites, not executions, so the header reports
    # the rewrite number rather than the raw execution ordinal.
    rewrite = report.rewrite or report.attempt
    header = (
        f"{report.gate}: {report.verdict.value}   "
        f"(attempt {min(rewrite, report.max_attempts)} of {report.max_attempts})"
    )
    out: list[str] = [header, ""]

    failures = report.failed_checks()
    if failures:
        out.append("FAILED CHECKS")
        out.append("")
        for check in failures:
            out.extend(_render_check(check))
        out.append("")

    warnings = report.warnings()
    if warnings:
        out.append("WARNINGS (not blocking, but the report must not ignore these)")
        out.append("")
        for check in warnings:
            # Same evidence renderers as failures: a warning the agent cannot
            # locate in the logs is a warning it will ignore.
            out.extend(_render_check(check))

    # The model writes the fixes when it produced grounded ones; the template is
    # the fallback, not the default. Either way the findings above are the
    # deterministic ones, verbatim — the model explains what to do, it does not
    # get to restate what happened.
    if report.generated_fixes:
        out.append("REQUIRED FIXES")
        out.extend(
            f"  {line}" if line.strip() else line
            for line in report.generated_fixes.splitlines()
        )
        out.append("")
    else:
        fixes = _required_fixes(failures)
        if fixes:
            out.append("REQUIRED FIXES")
            for i, fix in enumerate(fixes, 1):
                out.append(f"  {i}. {fix}")
            note = _fallback_reason(report) if failures else None
            if note:
                out.append("")
                out.append(f"  ({note})")
            out.append("")

    execution = report.execution
    if execution is not None:
        out.append(
            f"STDOUT: {execution.stdout_bytes:,} bytes captured in full "
            f"({execution.stdout_path})"
        )
        if execution.metrics:
            keys = ", ".join(sorted(execution.metrics))
            out.append(f"RECORDED: {keys}")
        if include_stdout_tail and failures:
            tail = _tail(execution.stdout_text(limit=100_000), _MAX_TAIL_LINES)
            if tail:
                out.append("")
                out.append("LAST LINES BEFORE FAILURE")
                out.extend(f"  | {line}" for line in tail)
    elif failures:
        out.append("The experiment was rejected before execution; nothing was run.")

    return "\n".join(out).rstrip() + "\n"


def render_summary(report: GateReport) -> str:
    """One line, for the run log."""
    if report.passed:
        warned = len(report.warnings())
        suffix = f" ({warned} warning{'s' if warned != 1 else ''})" if warned else ""
        return f"{report.gate}: PASS{suffix}"
    ids = ", ".join(c.id for c in report.failed_checks())
    return f"{report.gate}: FAIL [{ids}]"


# --------------------------------------------------------------------------- #
# per-check rendering
# --------------------------------------------------------------------------- #


def render_evidence(check: CheckResult) -> list[str]:
    """The check's evidence as display lines, or empty when it has no renderer.

    Public because the evidence bundle needs the same rows the feedback report
    shows: a warning the writer is told to disclose but cannot see is not a
    warning, it is an instruction to guess.
    """
    renderer = _EVIDENCE_RENDERERS.get(check.id)
    return renderer(check) if renderer else []


def _render_check(check: CheckResult) -> list[str]:
    lines = [f"  [{check.id}]", f"    {check.message}"]
    lines.extend(f"    {line}" for line in render_evidence(check))
    lines.append("")
    return lines


def _evidence_unbound(check: CheckResult) -> list[str]:
    out = []
    for row in check.evidence.get("names", [])[:_MAX_EVIDENCE_ROWS]:
        out.append(f"  {row['name']}  — read at line {row['lineno']} in {row['scope']}")
        if row.get("source_line"):
            out.append(f"    {row['lineno']} | {row['source_line']}")
    return out


def _evidence_exception(check: CheckResult) -> list[str]:
    ev = check.evidence
    if not ev.get("lineno"):
        return []
    where = f"line {ev['lineno']}"
    if ev.get("function"):
        where += f" in {ev['function']}"
    out = [f"  at {where}"]
    if ev.get("source_line"):
        out.append(f"    {ev['lineno']} | {ev['source_line'].strip()}")
    return out


def _evidence_literals(check: CheckResult) -> list[str]:
    out = []
    for row in check.evidence.get("literals", [])[:_MAX_EVIDENCE_ROWS]:
        out.append(f"  {row['key']} = {row['value']}  (line {row['lineno']})")
        if row.get("source_line"):
            out.append(f"    {row['lineno']} | {row['source_line'].strip()}")
    return out


def _evidence_missing_keys(check: CheckResult) -> list[str]:
    ev = check.evidence
    out = []
    if ev.get("missing"):
        out.append("  missing:  " + ", ".join(ev["missing"][:_MAX_EVIDENCE_ROWS]))
    if ev.get("recorded"):
        out.append("  recorded: " + ", ".join(ev["recorded"][:_MAX_EVIDENCE_ROWS]))
    return out


def _evidence_syntax(check: CheckResult) -> list[str]:
    ev = check.evidence
    if not ev.get("source_line"):
        return []
    return [f"  {ev.get('lineno')} | {ev['source_line'].strip()}"]


def _evidence_banned(check: CheckResult) -> list[str]:
    return [
        f"  {row['call']} at line {row['lineno']}"
        for row in check.evidence.get("calls", [])[:_MAX_EVIDENCE_ROWS]
    ]


def _evidence_shadowed(check: CheckResult) -> list[str]:
    return [
        f"  {row['name']} redefined as a {row['kind']} at line {row['lineno']}"
        + (f"\n    {row['lineno']} | {row['source_line'].strip()}"
           if row.get("source_line") else "")
        for row in check.evidence.get("shadowed", [])[:_MAX_EVIDENCE_ROWS]
    ]


def _evidence_log_signals(check: CheckResult) -> list[str]:
    out = []
    seen: set[str] = set()
    for row in check.evidence.get("findings", [])[:_MAX_EVIDENCE_ROWS]:
        if row["signal"] not in seen:
            seen.add(row["signal"])
            out.append(f"  {row['signal']}: {row['note']}")
        out.append(f"    {row['stream']}:{row['lineno']} | {row['line']}")
    return out


def _evidence_varied(check: CheckResult) -> list[str]:
    out = []
    for row in check.evidence.get("varied", [])[:_MAX_EVIDENCE_ROWS]:
        span = ""
        if row.get("min") is not None and row.get("max") is not None:
            span = f", ranged {row['min']} to {row['max']}"
        out.append(
            f"  {row['key']}: recorded {row['call_count']} times{span}; "
            f"registry holds {row['recorded_value']}"
        )
    return out


_EVIDENCE_RENDERERS = {
    "static.syntax_valid": _evidence_syntax,
    "static.no_unbound_names": _evidence_unbound,
    "static.no_banned_calls": _evidence_banned,
    "results.contract_not_shadowed": _evidence_shadowed,
    "exec.no_uncaught_exception": _evidence_exception,
    "results.values_computed": _evidence_literals,
    "results.expected_keys_present": _evidence_missing_keys,
    "logs.no_error_signals": _evidence_log_signals,
    # Same evidence shape, so model findings render exactly like pattern ones
    # and the agent never has to learn a second format.
    "logs.model_error_signals": _evidence_log_signals,
    "results.single_observation": _evidence_varied,
}


# --------------------------------------------------------------------------- #
# fix directives
# --------------------------------------------------------------------------- #

_FIXES = {
    "static.syntax_valid": "Fix the syntax error before submitting the code again.",
    "static.no_unbound_names": (
        "Bind every name listed above before it is read, or pass it in as a "
        "parameter. A name defined in an earlier version of the code is NOT "
        "available — each run starts from an empty namespace."
    ),
    "static.no_banned_calls": (
        "Remove the exit call. Let the program end by reaching the last line."
    ),
    "exec.exit_code_zero": "The process did not exit cleanly. Fix the failure above.",
    "exec.no_uncaught_exception": (
        "Fix the exception at the line shown. Do not wrap it in try/except to "
        "silence it — a suppressed failure still invalidates the results."
    ),
    "exec.completed_within_budget": (
        "Reduce the time complexity: fewer epochs, a smaller subset, or a "
        "cheaper model. The run was killed before it finished."
    ),
    "env.clean_namespace": (
        "The execution environment was not clean. Report this — it is a harness "
        "fault, not a fault in your code."
    ),
    "results.contract_present": (
        "Record every metric named in the plan with "
        "record_result(\"<key>\", <value>) — for example "
        "record_result(\"exp1.K2.test_acc\", test_acc). The paper can only "
        "cite values recorded this way."
    ),
    "results.expected_keys_present": (
        "Add a record_result call for each missing key listed above."
    ),
    "results.contract_not_shadowed": (
        "Delete your own definition of record_result / record_metadata. They "
        "are already available in your namespace. Values passed to a function "
        "you defined yourself are printed, not recorded, and the paper cannot "
        "cite them."
    ),
    "results.values_computed": (
        "Pass the variable holding the measured value, not a number you typed. "
        "record_result(\"k\", test_acc) is valid; record_result(\"k\", 0.816) "
        "is not."
    ),
    "results.values_finite": (
        "A metric is NaN or infinite. Check for division by zero, an empty "
        "evaluation split, or a diverged loss."
    ),
    "env.code_identity": (
        "The source that ran does not hash to the source submitted. Report this "
        "— it is a harness fault, not a fault in your code."
    ),
}


def _fallback_reason(report: GateReport) -> str | None:
    """Why this report carries the template, when it does.

    Three situations, and they are not the same thing:

    * no model configured — the template is simply what this deployment's
      report is, and saying anything would be noise;
    * the model could not be reached — the reader is entitled to know the
      report is thinner than it should be;
    * the model answered and its answer was rejected as ungrounded — worth
      saying plainly, because it names a specific fault in the generated
      text rather than an outage, and it is the one an operator can act on.

    An earlier version returned a bool and printed "the model was unavailable"
    for all three, which was wrong for the last and the most misleading of the
    three: it reported an outage that had not happened.
    """
    ungrounded = next(
        (
            c
            for c in report.checks
            if c.id == "report.fixes_grounded" and c.evidence.get("ungrounded")
        ),
        None,
    )
    if ungrounded is not None:
        count = len(ungrounded.evidence["ungrounded"])
        # Deliberately does NOT name the invented tokens. Echoing them here
        # would put the hallucinated identifier back in front of the agent and
        # defeat the grounding check that just removed it — the engineer could
        # still go looking for `dropout_rate`. The names are recorded in
        # report.fixes_grounded and gate1_report.json, where an operator
        # debugging the generator can read them and the agent cannot.
        return (
            f"standard guidance — the specific fixes written for this attempt "
            f"referred to {count} name{'s' if count != 1 else ''} this run does "
            f"not contain, so they were discarded"
        )
    if report.model_degraded:
        return (
            "standard guidance — the model that writes specific fixes could "
            "not be reached for this attempt"
        )
    return None


def _required_fixes(failures: list[CheckResult]) -> list[str]:
    seen: set[str] = set()
    fixes: list[str] = []
    for check in failures:
        fix = _FIXES.get(check.id)
        if fix and fix not in seen:
            seen.add(fix)
            fixes.append(fix)
    return fixes


def _tail(text: str, n: int) -> list[str]:
    """Last few non-blank lines, each clipped — one banner line of repeated text
    should not consume the agent's context window."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    out = []
    for line in lines[-n:]:
        if len(line) > _MAX_TAIL_LINE_CHARS:
            line = f"{line[:_MAX_TAIL_LINE_CHARS]}… (+{len(line) - _MAX_TAIL_LINE_CHARS} chars)"
        out.append(line)
    return out
