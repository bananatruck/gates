"""Gate 1 on, Gate 1 off — the same engineer, the same task, the same model.

This is the experiment the layer exists to justify. Two arms differing in one
thing: what decides whether an experiment succeeded.

    gated     Gate 1's deterministic verdict decides. The engineer receives the
              feedback report; the writer receives the verified registry plus
              untruncated output.

    ungated   Upstream's rule decides: search the returned stdout for
              "[CODE EXECUTION ERROR]", where the returned stdout is the first
              1,000 characters of a buffer that has the marker appended after
              the program's own output. The engineer receives that same 1,000
              characters.

The ungated arm is a reconstruction, not a mock. It runs the same sandboxed
runner, so the two arms differ only in the decision rule and the channel width —
which is the comparison the paper needs. What it deliberately does not
reconstruct is upstream's shared-namespace `exec`, because that defect is
independent of the channel and would confound the measurement.

What to read from the result: `false_success` is the headline. It counts
attempts the ungated arm accepted that Gate 1 rejected — runs that would have
reached the writing phase carrying nothing, or carrying numbers no execution
produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from gates import Gate1Config, run_experiment, run_gate1
from gates.adapters.agentlab import MLE_GATE_INSTRUCTIONS

from . import reward as legacy

ModelFn = Callable[[str, str], str]

_SYSTEM = (
    "You are an ML engineer agent. You write a single self-contained Python "
    "experiment, which is executed in a fresh process. Reply with the complete "
    "program and nothing else: no explanation, no markdown fences, no commentary."
)

#: Upstream's ceiling, and the whole of its evidence channel.
LEGACY_MAX_LEN = 1000


@dataclass
class Attempt:
    turn: int
    code: str
    accepted: bool
    #: What the other arm would have decided about this same execution.
    counterpart_would_accept: bool | None = None
    failed_checks: list[str] = field(default_factory=list)
    recorded: dict[str, Any] = field(default_factory=dict)
    stdout_bytes: int = 0
    exit_code: int | None = None
    crashed: bool = False


@dataclass
class ArmResult:
    arm: str
    attempts: list[Attempt] = field(default_factory=list)
    accepted_at: int | None = None

    @property
    def turns(self) -> int:
        return len(self.attempts)

    @property
    def accepted(self) -> bool:
        return self.accepted_at is not None

    @property
    def false_success(self) -> int:
        """Attempts this arm accepted that the other arm would have rejected.

        Zero by construction for the gated arm. For the ungated arm it is the
        number of runs that would have reached the writer on no evidence.
        """
        return sum(
            1
            for a in self.attempts
            if a.accepted and a.counterpart_would_accept is False
        )

    @property
    def accepted_a_crash(self) -> int:
        return sum(1 for a in self.attempts if a.accepted and a.crashed)

    @property
    def accepted_without_results(self) -> int:
        return sum(1 for a in self.attempts if a.accepted and not a.recorded)


def _ask(
    model: ModelFn,
    task: str,
    feedback: str | None,
    gated: bool,
    *,
    contract: bool | None = None,
) -> str:
    """Build the engineer's prompt.

    ``contract`` controls whether the results-contract instructions are
    included, and the choice is a real methodological fork rather than a knob:

    * ``None`` (default) follows the arm. The ungated arm gets no contract,
      which is faithful — upstream has none — but it makes "accepted a run that
      recorded nothing" true of the ungated arm *by construction*, so that
      column stops being evidence.
    * ``True`` gives both arms the same instructions, so the arms differ only in
      the decision rule and the channel width. Weaker headline, cleaner claim,
      and immune to the objection that the comparison was won by withholding
      instructions from one side.

    Report which one produced a given table.
    """
    include_contract = gated if contract is None else contract
    parts = [f"TASK\n{task}"]
    if include_contract:
        parts.append(MLE_GATE_INSTRUCTIONS)
    if feedback:
        parts.append(
            "Your previous submission was REJECTED. What follows is all you are "
            f"told about why. Fix it and resubmit the whole program.\n\n{feedback}"
        )
    text = model("\n\n".join(parts), _SYSTEM)
    return _strip_fences(str(text))


def _strip_fences(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    parts = t.split("```")
    return parts[1].split("\n", 1)[-1] if len(parts) > 1 else t


def run_gated(
    model: ModelFn,
    task: str,
    keys: tuple[str, ...],
    workdir: Path,
    *,
    max_turns: int = 3,
    timeout_s: int = 180,
    consult_model: ModelFn | None = None,
    contract: bool | None = None,
) -> ArmResult:
    result = ArmResult(arm="gated")
    config = Gate1Config(
        artifact_root=str(workdir),
        timeout_s=timeout_s,
        expected_keys=keys,
        max_attempts=max_turns,
        consult_model=consult_model,
        task_ref=task,
    )
    feedback = None
    for turn in range(1, max_turns + 1):
        code = _ask(model, task, feedback, gated=True, contract=contract)
        report = run_gate1(code, config, attempt=turn)
        execution = report.execution
        attempt = Attempt(
            turn=turn,
            code=code,
            accepted=report.passed,
            failed_checks=sorted(c.id for c in report.failed_checks()),
            recorded={k: m.value for k, m in report.metrics().items()},
            stdout_bytes=execution.stdout_bytes if execution else 0,
            exit_code=execution.exit_code if execution else None,
            crashed=bool(execution and execution.exception),
        )
        # Would upstream's rule have accepted this same execution?
        if execution is not None:
            attempt.counterpart_would_accept = not legacy.upstream_detects_failure(
                execution, LEGACY_MAX_LEN
            )
        result.attempts.append(attempt)
        if report.passed:
            result.accepted_at = turn
            break
        from gates.report import render_feedback

        feedback = render_feedback(report)
    return result


def run_ungated(
    model: ModelFn,
    task: str,
    workdir: Path,
    *,
    max_turns: int = 3,
    timeout_s: int = 180,
    contract: bool | None = None,
) -> ArmResult:
    """Upstream's loop: execute, slice to 1,000 chars, search for the marker."""
    result = ArmResult(arm="ungated")
    feedback = None
    for turn in range(1, max_turns + 1):
        code = _ask(model, task, feedback, gated=False, contract=contract)
        execution = run_experiment(
            code, workdir / f"attempt_{turn:02d}", timeout_s=timeout_s
        )
        view = legacy.legacy_view(execution, LEGACY_MAX_LEN)
        upstream_ok = legacy.LEGACY_MARKER not in view

        # What the gate would have said about the same execution, for the
        # counterfactual. Re-parsed from the same artifacts; nothing re-runs.
        gate_report = run_gate1(
            code,
            Gate1Config(
                artifact_root=str(workdir / "shadow_gate"),
                timeout_s=timeout_s,
                task_ref=task,
            ),
            attempt=turn,
        )

        attempt = Attempt(
            turn=turn,
            code=code,
            accepted=upstream_ok,
            counterpart_would_accept=gate_report.passed,
            failed_checks=sorted(c.id for c in gate_report.failed_checks()),
            recorded={k: m.value for k, m in gate_report.metrics().items()},
            stdout_bytes=execution.stdout_bytes,
            exit_code=execution.exit_code,
            crashed=execution.exception is not None,
        )
        result.attempts.append(attempt)
        if upstream_ok:
            result.accepted_at = turn
            break
        feedback = view  # upstream hands back exactly what it saw
    return result


def render(gated: ArmResult, ungated: ArmResult) -> str:
    out = ["GATE 1 ABLATION — same task, same model, same engineer", ""]
    header = f"  {'ARM':<9}{'TURNS':>6}{'ACCEPTED':>10}{'AT TURN':>9}{'CRASH OK':>10}{'NO RESULTS OK':>15}{'FALSE SUCCESS':>15}"
    out += [header, "  " + "-" * (len(header) - 2)]
    for arm in (gated, ungated):
        out.append(
            f"  {arm.arm:<9}{arm.turns:>6}{str(arm.accepted):>10}"
            f"{str(arm.accepted_at or '—'):>9}{arm.accepted_a_crash:>10}"
            f"{arm.accepted_without_results:>15}{arm.false_success:>15}"
        )
    out += ["", "  per attempt"]
    for arm in (gated, ungated):
        out.append(f"    {arm.arm}")
        for a in arm.attempts:
            verdict = "ACCEPT" if a.accepted else "reject"
            other = (
                "—"
                if a.counterpart_would_accept is None
                else ("accept" if a.counterpart_would_accept else "reject")
            )
            out.append(
                f"      turn {a.turn}: {verdict:<7} other_arm={other:<7}"
                f" exit={a.exit_code} stdout={a.stdout_bytes:,}B"
                f" recorded={list(a.recorded) or '[]'}"
                + (f" failed={a.failed_checks}" if a.failed_checks else "")
            )
    out += [
        "",
        "  FALSE SUCCESS is the headline: attempts this arm accepted that the",
        "  other would have rejected. For the ungated arm those are runs that",
        "  would have reached the writing phase carrying nothing measured.",
        "",
    ]
    return "\n".join(out)
