"""Wires Gate 1 into Agent-Researcher / Agent Laboratory's MLE solver.

The insertion point is between code execution and the reward model. Upstream,
``Replace.parse_command`` executed the code and decided success by searching the
(truncated) stdout for a marker string; ``get_score`` then asked an LLM for a
float and treated "a float parsed" as "the run worked". Here, execution goes
through the sandboxed runner, Gate 1 issues the verdict, and ``get_score``
ranks only what already passed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .. import (
    REGISTRY_FILENAME,
    Gate1Config,
    GateFailure,
    GateReport,
    Ledger,
    render_feedback,
    render_summary,
    run_gate1,
)

#: How much raw stdout to inline for the writing agent. Every citable number is
#: in the registry regardless, so this budget bounds context cost without
#: putting any value out of reach. Upstream's equivalent was 1000 characters,
#: and it held the whole experimental record.
STDOUT_BUDGET_CHARS = 40_000


MLE_GATE_INSTRUCTIONS = """
============= RESULTS RECORDING (REQUIRED) =============
Your code runs in a fresh Python process with an empty namespace. Nothing from a
previous version of your code exists. Every name you read must be bound by the
code you submit.

Every number the paper is allowed to report must be recorded with:

    record_result("<key>", <value>, unit="<unit>")

`record_result` is already available - do not import or define it. Use dotted
keys that name the experiment and the quantity, for example:

    record_result("exp1.K2.test_acc", test_acc, unit="ratio")
    record_result("exp2.sgc.wallclock_s", total_sgc_time, unit="seconds")

Pass the VARIABLE holding the measured value. A number typed directly into the
call is rejected: record_result("exp1.K2.test_acc", 0.816) fails the gate,
because a typed number is not a measurement.

Record the seed with record_metadata("seed", seed), along with any other
provenance that is not itself a result. A run with no declared seed cannot be
re-executed to confirm its own numbers, and the report has to say so.

If you record the same key more than once - once per epoch, for instance - the
registry keeps the LAST call, and every call is retained so the difference is
visible. Record the value you intend the paper to report, at the point it is
final. Do not record a running best and describe it as a final result.

Your code is checked before and after it runs. Unbound names, uncaught
exceptions, non-zero exit codes, missing declared keys and hardcoded values all
reject the submission and send it back to you with a report. Printing results is
still useful, but printed values are not citable - only recorded ones are.
"""


@dataclass
class GateContext:
    """Per-phase gate state: budget, ledger, and what has passed so far."""

    config: Gate1Config
    ledger: Ledger | None = None
    phase: str = "running experiments"
    reward_model: str | None = None
    #: Score gate-rejected attempts anyway, purely to record the disagreement.
    #: This is the paper's central evidence table; it costs one model call per
    #: rejection.
    shadow_reward_on_reject: bool = True

    #: Executions performed. Names artifact directories; not the budget.
    attempt: int = 0
    #: Agent-level rewrites rejected in a row. This is what the budget counts —
    #: a single rewrite may spend several executions on automated repair, and
    #: those must not eat the ML engineer's turns.
    consecutive_rejections: int = 0
    has_passing_attempt: bool = False
    last_report: GateReport | None = None
    history: list[GateReport] = field(default_factory=list)

    @property
    def budget_exhausted(self) -> bool:
        return self.consecutive_rejections >= self.config.max_attempts

    def check_can_continue(self) -> None:
        """Raise once the budget is spent and nothing ever passed.

        A run that never produced a valid experiment must not produce a paper.
        If something did pass, the caller falls back to it instead.
        """
        if self.budget_exhausted and not self.has_passing_attempt:
            raise GateFailure(
                gate="GATE 1 — EXECUTION VALIDITY",
                attempts=self.consecutive_rejections,
                report=self.last_report,
            )

    def note(self, report: GateReport) -> None:
        """Record one execution. Does not move the budget."""
        self.last_report = report
        self.history.append(report)
        if report.passed:
            self.has_passing_attempt = True

    def close_turn(self, passed: bool) -> None:
        """Close one ML engineer turn, however many executions it took."""
        self.consecutive_rejections = 0 if passed else self.consecutive_rejections + 1


@dataclass
class GatedExecution:
    """What the solver gets back in place of a raw stdout string."""

    report: GateReport
    feedback: str
    evidence_bundle: str
    code: str = ""

    @property
    def passed(self) -> bool:
        return self.report.passed

    @property
    def summary(self) -> str:
        return render_summary(self.report)

    def legacy_view(self, max_len: int = 1000) -> str:
        """Reconstruct exactly what upstream's ``execute_code`` would have returned.

        Upstream appended its error marker to the *end* of the capture buffer and
        then sliced ``[:MAX_LEN]``, so on any run that printed more than
        ``max_len`` characters the marker fell off and the crash became
        invisible. Reproducing that view lets the ledger answer the
        counterfactual — what would the reward model have scored, given the
        channel the scaffold actually had? — instead of guessing at it.
        """
        execution = self.report.execution
        if execution is None:
            return ""
        buffer = execution.stdout_text()
        exc = execution.exception
        if exc is not None:
            buffer += f"[CODE EXECUTION ERROR]: {exc.message}\n{exc.traceback}"
        return buffer[:max_len]


def make_context(
    *,
    research_dir: str = "./research_dir",
    phase: str = "running experiments",
    max_attempts: int = 3,
    timeout_s: int = 900,
    expected_keys: tuple[str, ...] | None = None,
    require_metrics: bool = True,
    num_classes: int | None = None,
    reward_model: str | None = None,
    task_ref: str | None = None,
) -> GateContext:
    """Build the gate context for one solver phase.

    ``task_ref`` is the plan or task text the experiment implements. Passing it
    closes the first link of each value's provenance chain; omitting it leaves
    that link recorded as unresolved rather than assumed.
    """
    artifact_root = os.path.join(research_dir, "gate_artifacts")
    config = Gate1Config(
        max_attempts=max_attempts,
        timeout_s=timeout_s,
        expected_keys=expected_keys,
        require_metrics=require_metrics,
        num_classes=num_classes,
        artifact_root=artifact_root,
        cwd=os.getcwd(),  # figures must land where papersolver looks for them
        task_ref=task_ref,
    )
    return GateContext(
        config=config,
        ledger=Ledger(os.path.join(artifact_root, "divergence.jsonl")),
        phase=phase,
        reward_model=reward_model,
    )


def gated_execute(code: str, context: GateContext) -> GatedExecution:
    """Execute ``code`` under Gate 1 and return the verdict plus its artifacts."""
    context.attempt += 1
    report = run_gate1(code, context.config, attempt=context.attempt)
    report.rewrite = context.consecutive_rejections + 1
    context.note(report)

    print(f"$$$$ {render_summary(report)}")

    return GatedExecution(
        report=report,
        feedback=render_feedback(report),
        evidence_bundle=build_evidence_bundle(report),
        code=code,
    )


def build_evidence_bundle(report: GateReport, budget: int = STDOUT_BUDGET_CHARS) -> str:
    """The evidence handed downstream in place of a 1000-character prefix.

    The verified registry comes first and is complete. Raw stdout follows, and
    is the only part subject to a budget — no citable value lives there that is
    not already in the registry above it.
    """
    if not report.passed:
        return f"[GATE 1 REJECTED THIS RUN]\n\n{render_feedback(report)}"

    metrics = report.metrics()
    execution_ = report.execution
    lines = [
        f"VERIFIED RESULTS — Gate 1 PASS (attempt {report.attempt})",
        "",
        "Every value below was recorded by record_result() during the run that",
        "produced the code above, and traced back to the line that computed it.",
        "These are the only numbers that may be reported.",
        "",
    ]
    if execution_ and execution_.run_id:
        lines += [
            f"  run {execution_.run_id}   code {(report.code_sha256 or '')[:12]}",
            "",
        ]
    if metrics:
        width = max(len(k) for k in metrics)
        for key in sorted(metrics):
            m = metrics[key]
            unit = f"  [{m.unit}]" if m.unit else ""
            where = f"  (experiment.py:{m.lineno})" if m.lineno else ""
            lines.append(f"  {key.ljust(width)} = {m.value}{unit}{where}")
    else:
        lines.append("  (none recorded)")

    execution = report.execution
    if execution and execution.metadata:
        lines += ["", "RUN METADATA"]
        lines += [f"  {k} = {v}" for k, v in sorted(execution.metadata.items())]

    warnings = report.warnings()
    if warnings:
        lines += ["", "WARNINGS — these must be stated in the report, not omitted"]
        lines += [f"  [{c.id}] {c.message}" for c in warnings]

    if report.artifact_dir:
        lines += [
            "",
            "PROVENANCE",
            f"  registry: {os.path.join(report.artifact_dir, REGISTRY_FILENAME)}",
        ]

    if execution:
        stdout = execution.stdout_text()
        lines += [
            "",
            f"FULL EXPERIMENT OUTPUT ({execution.stdout_bytes:,} bytes, untruncated "
            f"at {execution.stdout_path})",
            "",
            _fit(stdout, budget),
        ]
    return "\n".join(lines)


def record_divergence(
    context: GateContext,
    report: GateReport,
    reward_score: float | None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Log what the gate decided next to what the reward model said."""
    if context.ledger is None:
        return
    context.ledger.record_attempt(
        report,
        phase=context.phase,
        reward_score=reward_score,
        reward_model=context.reward_model,
        extra=extra,
    )


def _fit(text: str, budget: int) -> str:
    """Keep the head and the tail; say exactly how much was elided."""
    if len(text) <= budget:
        return text
    half = budget // 2
    elided = len(text) - budget
    return (
        f"{text[:half]}\n\n"
        f"    [... {elided:,} characters elided from the middle of stdout. "
        f"All recorded results appear in the registry above; the complete "
        f"output is on disk ...]\n\n"
        f"{text[-half:]}"
    )
