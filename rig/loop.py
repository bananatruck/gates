"""The Gate 1 feedback loop, driven end to end without a model.

This is the loop the adapter contract describes, run for real: the engineer
submits code, Gate 1 issues a verdict, a rejection renders a feedback report and
costs one rewrite, and the budget is spent in agent turns rather than
executions. Nothing here is a mock — ``gated_execute``, ``run_gate1``,
``render_feedback`` and the ledger are the shipped code paths. Only the engineer
is scripted, which is what makes the loop cheap enough to run on every change.

Plug a real engineer in by implementing :class:`Engineer`; the loop does not care
where the code came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from gates import GateFailure, GateReport
from gates.adapters.agentlab import (
    GateContext,
    build_evidence_bundle,
    gated_execute,
    make_context,
    record_divergence,
)

from . import reward as legacy
from .scenarios import Scenario, Step

#: Refuse to spin forever if an engineer keeps returning code and the budget
#: never advances. Bounded loops are a design invariant; this enforces it on the
#: rig as well as on the gate.
MAX_TURNS_GUARD = 24

Observer = Callable[[str, dict[str, Any]], None]

#: A model that scores an attempt the way upstream's ``get_score`` did. Takes
#: the 1000-character view the scaffold would have had and returns a float in
#: [0, 1]. Left unset by default: inventing a plausible number here would put
#: fiction in the ledger the paper is generated from.
RewardFn = Callable[[str], float]


class Engineer(Protocol):
    """Whatever writes the code. A script here, a model in the real loop."""

    def turn(self, feedback: str | None, turn_index: int) -> Sequence[Step] | None:
        """Return the executions this turn will spend, or ``None`` to give up.

        More than one step means the host scaffold ran an inner automated-repair
        loop inside a single agent turn. The loop charges the turn once.
        """


class ScriptedEngineer:
    """Replays a scenario. Reads the feedback and ignores it, on purpose —
    a fixed script makes the gate's behaviour the only variable."""

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.feedback_seen: list[str] = []

    def turn(self, feedback: str | None, turn_index: int) -> Sequence[Step] | None:
        if feedback is not None:
            self.feedback_seen.append(feedback)
        if turn_index >= len(self.scenario.turns):
            return None
        return self.scenario.turns[turn_index].steps


# --------------------------------------------------------------------------- #
# what the loop reports back
# --------------------------------------------------------------------------- #


@dataclass
class ExecutionOutcome:
    """One execution: the gate's verdict, and what upstream would have made of it."""

    label: str
    attempt: int
    report: GateReport
    feedback: str
    #: Did upstream's substring test see the failure through a 1000-char channel?
    #: ``None`` when the question does not arise — the run did not fail.
    upstream_detected_failure: bool | None = None
    #: Channel width at which the crash marker becomes visible. ``None`` if the
    #: run did not crash.
    marker_visible_at: int | None = None
    #: True when the counterfactual required executing code the gate rejected
    #: statically — upstream had no static tier, so it would have paid for it.
    counterfactual_executed: bool = False

    @property
    def passed(self) -> bool:
        return self.report.passed

    @property
    def failed_check_ids(self) -> set[str]:
        return {c.id for c in self.report.failed_checks()}

    @property
    def warned_check_ids(self) -> set[str]:
        return {c.id for c in self.report.warnings()}

    @property
    def upstream_blind(self) -> bool:
        """Gate rejected it; upstream's detector would have called it a success."""
        return not self.passed and self.upstream_detected_failure is False


@dataclass
class TurnOutcome:
    index: int
    executions: list[ExecutionOutcome] = field(default_factory=list)
    passed: bool = False
    #: Budget consumed after this turn closed.
    rejections_after: int = 0


@dataclass
class LoopOutcome:
    scenario: str
    turns: list[TurnOutcome] = field(default_factory=list)
    #: "pass" — something was admitted. "gate_failure" — budget spent with
    #: nothing admitted, so no paper is produced. "no_pass" — the engineer gave
    #: up before the budget did.
    outcome: str = "no_pass"
    gate_failure: GateFailure | None = None
    evidence_bundle: str | None = None
    fell_back: bool = False
    ledger_path: str | None = None
    artifact_root: str | None = None
    #: Whether the upstream-channel reconstruction ran. When it did not, the
    #: scenario's `expect_upstream_blind_turns` describes something that was
    #: never measured, and asserting on it would report a mismatch that is an
    #: artefact of the switch rather than a finding.
    counterfactual: bool = True

    @property
    def executions(self) -> list[ExecutionOutcome]:
        return [e for t in self.turns for e in t.executions]

    @property
    def model_cost(self) -> dict[str, Any]:
        """What the LLM layer spent across the whole loop.

        The gate's standing claim is that a verdict costs zero model calls and
        that rejection is cheap. The layer does not change the first and must not
        wreck the second, so the total is reported per loop and held to a ceiling
        by test rather than assumed to be small.
        """
        totals = {
            "calls": 0,
            "failures": 0,
            "latency_s": 0.0,
            "prompt_chars": 0,
            "completion_chars": 0,
        }
        for execution in self.executions:
            spent = execution.report.model
            if not spent:
                continue
            for key in totals:
                totals[key] += spent.get(key, 0)
        totals["latency_s"] = round(totals["latency_s"], 3)
        totals["degraded"] = any(
            e.report.model_degraded for e in self.executions
        )
        totals["calls_per_execution"] = (
            round(totals["calls"] / len(self.executions), 2)
            if self.executions
            else 0.0
        )
        return totals

    @property
    def turns_used(self) -> int:
        return len(self.turns)

    @property
    def upstream_blind_turns(self) -> tuple[int, ...]:
        """Turns the gate rejected that upstream's detector would have passed."""
        return tuple(
            t.index
            for t in self.turns
            if not t.passed and any(e.upstream_blind for e in t.executions)
        )


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #


def run_loop(
    scenario: Scenario,
    *,
    workdir: str | Path,
    engineer: Engineer | None = None,
    reward_fn: RewardFn | None = None,
    counterfactual: bool = True,
    timeout_s: int = 60,
    observer: Observer | None = None,
    consult_model: Any = None,
) -> LoopOutcome:
    """Play one scenario against the real gate and return what happened."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    engineer = engineer or ScriptedEngineer(scenario)

    context = make_context(
        research_dir=str(workdir),
        max_attempts=scenario.max_attempts,
        timeout_s=timeout_s,
        expected_keys=scenario.expected_keys,
        num_classes=scenario.num_classes,
        task_ref=scenario.summary,
        consult_model=consult_model,
    )
    outcome = LoopOutcome(
        scenario=scenario.name,
        artifact_root=context.config.artifact_root,
        ledger_path=str(context.ledger.path) if context.ledger else None,
        counterfactual=counterfactual,
    )
    emit = observer or (lambda event, payload: None)
    emit("loop_start", {"scenario": scenario, "context": context})

    feedback: str | None = None

    for turn_index in range(MAX_TURNS_GUARD):
        steps = engineer.turn(feedback, turn_index)
        if steps is None:
            break

        turn = TurnOutcome(index=turn_index)
        outcome.turns.append(turn)
        emit("turn_start", {"turn": turn, "steps": steps})

        for step in steps:
            gated = gated_execute(step.code, context)
            feedback = gated.feedback

            execution_outcome = _observe_upstream(
                step=step,
                report=gated.report,
                workdir=workdir,
                counterfactual=counterfactual,
                timeout_s=timeout_s,
                feedback=gated.feedback,
            )
            turn.executions.append(execution_outcome)

            record_divergence(
                context,
                gated.report,
                reward_score=reward_fn(gated.legacy_view()) if reward_fn else None,
                extra={
                    "turn": turn_index,
                    "step": step.label,
                    "scenario": scenario.name,
                    "upstream_detected_failure": (
                        execution_outcome.upstream_detected_failure
                    ),
                    "marker_visible_at": execution_outcome.marker_visible_at,
                    "counterfactual_executed": (
                        execution_outcome.counterfactual_executed
                    ),
                },
            )
            emit("execution", {"outcome": execution_outcome, "gated": gated})

            if gated.passed:
                turn.passed = True
                outcome.evidence_bundle = build_evidence_bundle(gated.report)
                break  # an inner repair loop stops once something works

        context.close_turn(turn.passed)
        turn.rejections_after = context.consecutive_rejections
        emit("turn_end", {"turn": turn, "context": context})

        if turn.passed and not scenario.continue_after_pass:
            break

        if not turn.passed:
            try:
                context.check_can_continue()
            except GateFailure as exc:
                outcome.gate_failure = exc
                outcome.outcome = "gate_failure"
                emit("gate_failure", {"error": exc})
                return outcome
            if context.budget_exhausted:
                # Budget spent, but an earlier attempt passed: fall back to it
                # rather than failing the phase.
                outcome.fell_back = True
                break

    outcome.outcome = "pass" if context.has_passing_attempt else "no_pass"
    if outcome.evidence_bundle is None and context.has_passing_attempt:
        last_pass = next(r for r in reversed(context.history) if r.passed)
        outcome.evidence_bundle = build_evidence_bundle(last_pass)
    emit("loop_end", {"outcome": outcome, "context": context})
    return outcome


def _observe_upstream(
    *,
    step: Step,
    report: GateReport,
    workdir: Path,
    counterfactual: bool,
    timeout_s: int,
    feedback: str,
) -> ExecutionOutcome:
    """Answer "would upstream have noticed?" for one execution.

    For an attempt that executed, the answer comes from its own capture. For one
    the static tier rejected before execution, it requires actually running the
    code — upstream had no static tier and would have paid for the run. Both
    cases are recorded; the second is flagged, because it is a reconstruction.
    """
    result = ExecutionOutcome(
        label=step.label,
        attempt=report.attempt,
        report=report,
        feedback=feedback,
    )
    if report.passed or not counterfactual:
        return result

    execution = report.execution
    if execution is None:
        shadow_dir = workdir / "upstream_counterfactual" / f"attempt_{report.attempt:02d}"
        execution = legacy.shadow_execute(
            step.code, shadow_dir, timeout_s=timeout_s, cwd=str(workdir)
        )
        result.counterfactual_executed = True

    result.upstream_detected_failure = legacy.upstream_detects_failure(execution)
    result.marker_visible_at = legacy.marker_survives_at(execution)
    return result


# --------------------------------------------------------------------------- #
# checking a run against what the scenario claims
# --------------------------------------------------------------------------- #


def check_expectations(scenario: Scenario, outcome: LoopOutcome) -> list[str]:
    """Every way this run departed from what the scenario says Gate 1 does.

    Empty list means the loop behaved exactly as documented.
    """
    problems: list[str] = []

    if outcome.outcome != scenario.expect_outcome:
        problems.append(
            f"outcome was {outcome.outcome!r}, expected {scenario.expect_outcome!r}"
        )
    if outcome.turns_used != scenario.expect_turns:
        problems.append(
            f"used {outcome.turns_used} engineer turn(s), "
            f"expected {scenario.expect_turns}"
        )

    for turn, spec in zip(outcome.turns, scenario.turns):
        for execution, step in zip(turn.executions, spec.steps):
            where = f"turn {turn.index + 1} / {step.label!r}"
            if step.expect_pass and not execution.passed:
                problems.append(
                    f"{where}: expected PASS, got FAIL "
                    f"[{', '.join(sorted(execution.failed_check_ids))}]"
                )
            if not step.expect_pass and execution.passed:
                problems.append(f"{where}: expected FAIL, got PASS")

            missing = set(step.expect_fail) - execution.failed_check_ids
            if missing:
                problems.append(
                    f"{where}: expected failing check(s) {sorted(missing)}, "
                    f"got [{', '.join(sorted(execution.failed_check_ids))}]"
                )
            missing_warn = set(step.expect_warn) - execution.warned_check_ids
            if missing_warn:
                problems.append(
                    f"{where}: expected warning(s) {sorted(missing_warn)}, "
                    f"got [{', '.join(sorted(execution.warned_check_ids))}]"
                )

    if scenario.expect_upstream_blind_turns and outcome.counterfactual:
        actual = outcome.upstream_blind_turns
        expected = scenario.expect_upstream_blind_turns
        if actual != expected:
            problems.append(
                f"upstream's detector was blind on turns {list(actual)}, "
                f"expected {list(expected)}"
            )

    if scenario.expect_outcome == "gate_failure" and outcome.gate_failure is None:
        problems.append("expected GateFailure to be raised; it was not")
    if scenario.expect_outcome == "pass" and outcome.evidence_bundle is None:
        problems.append("passed, but no evidence bundle was produced")

    return problems
