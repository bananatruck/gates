"""The gate loops every host shares, and the switch that says which gates run.

A host adapter builds the contexts, because only the host knows its phases,
paths and declarations. Everything past that point is the same for every host:
run the gate, hand a rejection back as feedback, count the turn, stop on the
budget. It lives here so a second host binds to it instead of copying it.

Nothing here opens a socket or imports a host. The arXiv resolver lives in
``adapters/arxiv.py`` for that reason (D41).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .errors import GateError
from .gate3 import RENDERED_FILENAME
from .report import render_evidence
from . import (
    REGISTRY_FILENAME,
    Gate1Config,
    Gate2Config,
    Gate3Config,
    GateFailure,
    GateReport,
    Ledger,
    build_registry,
    render_feedback,
    render_summary,
    run_gate1,
    run_gate2,
    run_gate3,
    unresolved_discrepancies,
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


#: Gate 3's counterpart to MLE_GATE_INSTRUCTIONS, for the paper writer (D31).
#: A test checks every token shown here is one the gate's own patterns read.
REPORT_GATE_INSTRUCTIONS = r"""
============= RESULT CITATION (REQUIRED) =============
Every number that reports a result of this study must be written as a token:

    \result{<key>}

using a key from the verified results, for example \result{exp1.K2.test_acc}.
The renderer replaces each token with the value exactly as it was measured.

Do not type a result number yourself, and do not round one. A number typed
into a findings section (abstract, results, discussion, conclusion) is
rejected, because a number with no key was never measured.

Every Results section must cite at least one recorded value. Describing
results without numbers is rejected too.

If the verification layer declared limitations, write \limitations{} where the
paper discusses its limitations. The renderer inserts them word for word. Do
not paraphrase them or leave them out.

Your manuscript is checked before it is accepted. A rejected manuscript comes
back to you with a report naming what to change.
"""


@dataclass
class GateContext:
    """Per-phase gate state: budget, ledger, and what has passed so far."""

    config: Gate1Config | Gate2Config | Gate3Config
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
                gate=self.last_report.gate if self.last_report else "GATE",
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

    report: GateReport | None
    feedback: str
    evidence_bundle: str
    code: str = ""
    #: Set only on the bypass path, where there is no report to ask.
    ungated_passed: bool | None = None

    @property
    def passed(self) -> bool:
        if self.report is None:
            # Bypass: upstream's rule, which is a substring search over the
            # already-truncated view. Not a stand-in for it — the same test.
            return bool(self.ungated_passed)
        return self.report.passed

    @property
    def summary(self) -> str:
        if self.report is None:
            return f"gate bypassed — {'accepted' if self.passed else 'rejected'}"
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
        if self.report is None:
            # The gate-off control already holds exactly this view.
            return self.evidence_bundle
        execution = self.report.execution
        if execution is None:
            return ""
        buffer = execution.stdout_text()
        exc = execution.exception
        if exc is not None:
            buffer += f"[CODE EXECUTION ERROR]: {exc.message}\n{exc.traceback}"
        return buffer[:max_len]


def gate_level() -> int:
    """Which gates run, read from ``GATES_LEVEL``: 0 none, 1 Gate 1, 2 Gates 1
    and 2, 3 all three (D58).

    Cumulative because each gate reads what the one before it produced: Gate 2
    reviews Gate 1's registry, and Gate 3 judges a manuscript against it. So
    "Gate 2 without Gate 1" is not an arm, and no setting can ask for it.

    The switch exists so the arms of a full-workflow comparison differ in the
    gates and in nothing else. Checking out an older branch instead would also
    change the model plumbing, the rate-limit backoff and the prompts, and any
    difference in the papers could then be attributed to those.

    ``GATES_GATE1=off`` predates this and still means 0: the published Gate 1
    evidence and the ablation runner that produced it set it (D53). Unset, every
    gate is on.
    """
    level = os.environ.get("GATES_LEVEL")
    legacy = os.environ.get("GATES_GATE1")
    if level is None:
        legacy_off = (legacy or "on").strip().lower() in {"off", "0", "false", "no"}
        return 0 if legacy_off else 3
    if legacy is not None:
        raise GateError("set GATES_LEVEL or GATES_GATE1, not both")
    if level.strip() not in {"0", "1", "2", "3"}:
        raise GateError(f"GATES_LEVEL must be 0, 1, 2 or 3, not {level!r}")
    return int(level)


def gate1_enabled() -> bool:
    """Whether Gate 1 arbitrates. A host's phases ask this or :func:`gate_level`."""
    return gate_level() >= 1


def require_gate(gate: int) -> None:
    """Raise unless ``gate`` is open at the current :func:`gate_level`.

    Every entry point calls this before it asks the agent for anything, so a
    closed gate costs no turn and cannot run on an input the level below it did
    not produce.
    """
    level = gate_level()
    if level < gate:
        raise GateError(f"Gate {gate} is closed at GATES_LEVEL={level}; it opens at {gate}")


def gated_execute(code: str, context: GateContext) -> GatedExecution:
    """Execute ``code`` under Gate 1 and return the verdict plus its artifacts."""
    require_gate(1)
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


def gated_review(registry: dict[str, Any], context: GateContext) -> GatedExecution:
    """Review a Gate 1 registry under Gate 2 and return the verdict.

    The mirror of :func:`gated_execute` one phase later. The subject is the
    registry Gate 1 wrote rather than source code, and the budget is spent the
    same way: ``rewrite`` is set from the context so the feedback header counts
    agent turns rather than reporting "attempt 1" forever.

    The caller does **not** call ``check_can_continue`` after this. Gate 2's
    policy on a spent budget is to proceed with the discrepancies declared, so
    the loop keeps going and the limitations travel forward in the bundle.
    """
    require_gate(2)
    context.attempt += 1
    report = run_gate2(registry, context.config, attempt=context.attempt)
    report.rewrite = context.consecutive_rejections + 1
    context.note(report)

    print(f"$$$$ {render_summary(report)}")

    return GatedExecution(
        report=report,
        feedback=render_feedback(report),
        evidence_bundle=declared_limitations(report),
    )


def declared_limitations(report: GateReport) -> str:
    """What the manuscript has to disclose, whether or not Gate 2 passed.

    Gate 2 proceeds on exhaustion, so its findings must travel into the writing
    phase rather than stopping the run. An empty result means there is nothing
    to declare, not that nothing was checked - the report says which checks ran.
    """
    limitations = unresolved_discrepancies(report)
    if not limitations:
        return ""
    lines = ["DECLARED LIMITATIONS", ""]
    lines.extend(f"  - {item}" for item in limitations)
    return "\n".join(lines) + "\n"


@dataclass
class ReviewOutcome:
    """How Gate 2's feedback loop ended, and what the writer must disclose."""

    #: "pass" - a registry was admitted. "proceeded" - a budget was spent, Gate
    #: 2's or Gate 1's after a review, and the run continues with the last
    #: review's discrepancies declared. "no_pass" - ``revise`` stopped before
    #: either budget did.
    outcome: str = "no_pass"
    #: From the last review, on every exit. Empty means nothing to declare, not
    #: that nothing was checked.
    declared: str = ""
    #: The registry of the run Gate 2 last reviewed: what the writer cites and
    #: Gate 3 checks against. ``None`` when nothing was reviewed.
    registry: dict[str, Any] | None = None
    #: The Gate 1 run that registry came from, ``first`` included. Its ``code``
    #: and ``evidence_bundle`` are what the writer describes; the code the phase
    #: started with is stale once a revision is reviewed.
    run: GatedExecution | None = None
    reviews: list[GatedExecution] = field(default_factory=list)
    #: Every Gate 1 run this loop made, in order, passed or not. ``first`` is
    #: not among them: the loop reviewed it but did not run it.
    executions: list[GatedExecution] = field(default_factory=list)


def review_loop(
    context: GateContext,
    revise: Callable[[str | None], str | None],
    *,
    gate1: GateContext,
    first: GatedExecution | None = None,
    extra: dict[str, Any] | None = None,
) -> ReviewOutcome:
    """Gate 2's feedback loop, tier C: the one call site a host needs.

    ``revise(feedback)`` returns the next version of the experiment's code, or
    ``None`` to stop. Each version runs under Gate 1 on ``gate1`` first, and
    Gate 2 reviews only the registry Gate 1 built from that run, so a fix cannot
    reach Gate 2 without running (F12). A Gate 1 rejection sends Gate 1's report
    back and costs a Gate 1 turn, not a Gate 2 one.

    ``first`` is the Gate 1 pass the host already holds from its experiment
    phase. It is reviewed without running again, and ``revise`` is first called
    with that review's feedback. Without it, ``revise`` is first called with
    ``None`` and its code is the first submission.

    Bounded by both budgets. A spent Gate 2 budget does not raise (`CLAUDE.md`
    §4). A spent Gate 1 budget raises ``GateFailure`` if no revision ever ran
    clean, and otherwise ends the loop with the last review declared.
    """
    require_gate(2)
    if first is not None and (first.report is None or not first.passed):
        raise GateError("first must be a run Gate 1 passed; Gate 1's rejection stands")
    result = ReviewOutcome()
    feedback: str | None = None
    executed = first
    while executed is not None or (code := revise(feedback)) is not None:
        if executed is None:
            executed = gated_execute(code, gate1)
            result.executions.append(executed)
            # review_turn, not turn: loop_summary reads turn and counts Gate 2 only.
            record_divergence(
                gate1,
                executed.report,
                reward_score=None,
                extra={"review_turn": len(result.reviews), **(extra or {})},
            )
            gate1.close_turn(executed.passed)
            if not executed.passed:
                # Nothing ever passed: Gate 1 raises, as it does outside this loop.
                gate1.check_can_continue()
                if gate1.budget_exhausted:
                    # Something passed and Gate 2 reviewed it, so that review's
                    # discrepancies stand declared, as on a spent Gate 2 budget.
                    result.outcome = "proceeded" if result.reviews else "no_pass"
                    break
                feedback = executed.feedback
                executed = None
                continue
        registry = build_registry(executed.report, task_ref=gate1.config.task_ref)
        run, executed = executed, None
        reviewed = gated_review(registry, context)
        record_divergence(
            context,
            reviewed.report,
            reward_score=None,
            extra={
                "turn": len(result.reviews),
                "max_attempts": context.config.max_attempts,
                **(extra or {}),
            },
        )
        context.close_turn(reviewed.passed)
        result.reviews.append(reviewed)
        result.registry, result.run = registry, run
        # Set on every turn, so a revise that gives up still hands the writer
        # the discrepancies it was sent (F4).
        result.declared = reviewed.evidence_bundle
        if reviewed.passed:
            result.outcome = "pass"
            break
        if context.budget_exhausted:
            result.outcome = "proceeded"
            break
        feedback = reviewed.feedback
    return result


def gated_report(
    source: str,
    registry: dict[str, Any],
    context: GateContext,
    *,
    declared: str = "",
    retrieved: Iterable[str] | None = None,
) -> GatedExecution:
    """Judge one manuscript under Gate 3 and return the verdict.

    The mirror of :func:`gated_execute` one phase later (D30). ``source`` is the
    writer's output with its ``\\result{}`` and ``\\limitations{}`` tokens
    intact. ``declared`` is ``ReviewOutcome.declared``, and ``retrieved`` the
    paper ids the host's searches returned (:func:`retrieved_arxiv_ids`). The
    evidence bundle is
    the render Gate 3 wrote to disk, so what a host publishes is byte for byte
    what the gate judged.
    """
    require_gate(3)
    context.attempt += 1
    report = run_gate3(
        source,
        registry,
        context.config,
        attempt=context.attempt,
        declared=declared,
        retrieved=retrieved,
    )
    report.rewrite = context.consecutive_rejections + 1
    context.note(report)

    print(f"$$$$ {render_summary(report)}")

    return GatedExecution(
        report=report,
        feedback=render_feedback(report),
        evidence_bundle=(Path(report.artifact_dir) / RENDERED_FILENAME).read_text(
            encoding="utf-8"
        ),
        code=source,
    )


@dataclass
class ReportOutcome:
    """How Gate 3's feedback loop ended. No "proceeded": a spent budget raises."""

    #: "pass" - a manuscript was admitted. "no_pass" - ``write`` stopped first.
    outcome: str = "no_pass"
    #: The admitted manuscript, rendered. The only text a host should publish,
    #: and ``None`` unless Gate 3 passed it.
    manuscript: str | None = None
    reports: list[GatedExecution] = field(default_factory=list)


def report_loop(
    context: GateContext,
    write: Callable[[str | None], str | None],
    *,
    registry: dict[str, Any],
    declared: str = "",
    retrieved: Callable[[], Iterable[str]] | None = None,
    extra: dict[str, Any] | None = None,
) -> ReportOutcome:
    """Gate 3's feedback loop: the one call site a host's writing phase needs.

    ``write(feedback)`` returns the next manuscript with its ``\\result{}``
    tokens intact, or ``None`` to stop, and is first called with ``None``.
    ``registry`` is the one the writer cites: ``ReviewOutcome.registry`` when
    Gate 2 ran, Gate 1's otherwise. ``declared`` is ``ReviewOutcome.declared``,
    the limitations the manuscript must state; empty means none. ``retrieved``
    returns the paper ids the host has retrieved so far and is called after each
    ``write``, because the reference host searches while it writes (D32);
    ``None`` leaves citations unchecked. Taking both
    rather than a ``ReviewOutcome`` keeps Gate 3 usable by a host that does not
    run Gate 2.

    A spent budget raises ``GateFailure`` (`CLAUDE.md` §4): an unverifiable
    manuscript is not emitted.
    """
    require_gate(3)
    if not registry.get("citable"):
        # run_gate3 refuses it too, but only after the writer spent a turn.
        raise GateError("Gate 1 did not pass, so there is nothing a manuscript may cite")
    result = ReportOutcome()
    feedback: str | None = None
    while (source := write(feedback)) is not None:
        written = gated_report(
            source,
            registry,
            context,
            declared=declared,
            retrieved=None if retrieved is None else retrieved(),
        )
        record_divergence(
            context,
            written.report,
            reward_score=None,
            extra={
                "turn": len(result.reports),
                "max_attempts": context.config.max_attempts,
                **(extra or {}),
            },
        )
        context.close_turn(written.passed)
        result.reports.append(written)
        if written.passed:
            result.outcome = "pass"
            result.manuscript = written.evidence_bundle
            break
        # Nothing earlier passed, or the loop would have ended, so this raises
        # whenever the budget is spent.
        context.check_can_continue()
        feedback = written.feedback
    return result


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
        for check in warnings:
            lines.append(f"  [{check.id}] {check.message}")
            # The evidence, not only the count. Telling the writer that "1 line
            # reports trouble the run continued past" and then withholding the
            # line asks it to disclose something it cannot see — which is this
            # layer's own failure mode, reproduced at its exit. render_feedback
            # already renders these for the engineer; the writer needs them at
            # least as much, because it is the one making the claim.
            lines += [f"    {row}" for row in render_evidence(check)]

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
