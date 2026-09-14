"""Run Gate 2's feedback loop, tier C, and show it working.

    python -m rig.gate2_loop                         # every scenario
    python -m rig.gate2_loop divergence-exhausts     # one
    python -m rig.gate2_loop --list
    python -m rig.gate2_loop --quiet --json          # for CI

Tier C adds no check. It is the loop tiers A and B run inside, with the shape of
Gate 1's (`rig/loop.py`): the engineer submits, the gate issues a verdict, a
rejection renders a feedback report and costs one agent turn, and the budget
lives in the adapter's ``GateContext``, never in the gate.

One deliberate difference, `CLAUDE.md` §4. A spent budget does not raise. The run
proceeds, and every discrepancy Gate 2 could not get resolved is handed to the
writing phase as a declared limitation, because a genuine novel result must not
be blocked forever and an unresolved one must not be dropped.

Exits non-zero if any scenario departs from what it documents. No model, no API
key, nothing executed.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gates import GateReport  # noqa: E402
from gates.adapters.agentlab import make_review_context, review_loop  # noqa: E402

from rig.gate2_scenarios import SCENARIOS, Scenario, Turn  # noqa: E402


class Engineer(Protocol):
    """Whatever produces the next registry. A script here, a re-run in a host."""

    def turn(self, feedback: str | None, turn_index: int) -> Turn | None:
        """Return this turn's submission, or ``None`` to give up."""


class ScriptedEngineer:
    """Replays a scenario. Reads the feedback and ignores it, on purpose, so the
    gate's behaviour is the only variable."""

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.feedback_seen: list[str] = []

    def turn(self, feedback: str | None, turn_index: int) -> Turn | None:
        if feedback is not None:
            self.feedback_seen.append(feedback)
        if turn_index >= len(self.scenario.turns):
            return None
        return self.scenario.turns[turn_index]


@dataclass
class TurnOutcome:
    index: int
    label: str
    report: GateReport
    feedback: str
    #: Consecutive rejections after this turn closed.
    rejections_after: int = 0

    @property
    def passed(self) -> bool:
        return self.report.passed


@dataclass
class LoopOutcome:
    scenario: str
    turns: list[TurnOutcome] = field(default_factory=list)
    #: "pass" - a registry was admitted. "proceeded" - the budget was spent and
    #: the run continues with its discrepancies declared. "no_pass" - the
    #: engineer gave up before the budget did.
    outcome: str = "no_pass"
    #: What the writing phase must disclose, from the review that ended the loop.
    #: Empty means nothing to declare, not that nothing was checked.
    declared: str = ""
    ledger_path: str | None = None

    @property
    def turns_used(self) -> int:
        return len(self.turns)


def run_gate2_loop(
    scenario: Scenario,
    *,
    workdir: str | Path,
    engineer: Engineer | None = None,
) -> LoopOutcome:
    """Play one scenario against the real Gate 2 and return what happened.

    Bounded by the budget rather than by a guard: every turn either passes and
    ends the loop or adds a consecutive rejection, so at most ``max_attempts``
    turns run.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    engineer = engineer or ScriptedEngineer(scenario)

    context = make_review_context(
        research_dir=str(workdir),
        max_attempts=scenario.max_attempts,
        relations=scenario.relations,
        plan_fields=scenario.plan_fields,
    )
    outcome = LoopOutcome(
        scenario=scenario.name,
        ledger_path=str(context.ledger.path) if context.ledger else None,
    )

    submitted: list[Turn] = []

    def revise(feedback: str | None) -> dict[str, Any] | None:
        turn = engineer.turn(feedback, len(submitted))
        if turn is None:
            return None
        submitted.append(turn)
        return turn.registry()

    reviewed = review_loop(context, revise, extra={"scenario": scenario.name})
    outcome.outcome = reviewed.outcome
    outcome.declared = reviewed.declared
    for index, (turn, review) in enumerate(zip(submitted, reviewed.reviews)):
        outcome.turns.append(
            TurnOutcome(
                index=index,
                label=turn.label,
                report=review.report,
                feedback=review.feedback,
                # rewrite was the rejection count plus one when this turn was
                # reviewed, so it is the count after a rejection closes it.
                rejections_after=0 if review.passed else review.report.rewrite,
            )
        )

    return outcome


def check_expectations(scenario: Scenario, outcome: LoopOutcome) -> list[str]:
    """Every way this run departed from what the scenario documents."""
    problems: list[str] = []

    if outcome.outcome != scenario.expect_outcome:
        problems.append(
            f"outcome was {outcome.outcome!r}, expected {scenario.expect_outcome!r}"
        )
    if outcome.turns_used != scenario.expect_turns:
        problems.append(
            f"used {outcome.turns_used} turn(s), expected {scenario.expect_turns}"
        )
    if outcome.outcome == "proceeded" and outcome.turns_used != scenario.max_attempts:
        problems.append(
            f"proceeded after {outcome.turns_used} turn(s) on a budget of "
            f"{scenario.max_attempts}"
        )

    for turn, spec in zip(outcome.turns, scenario.turns):
        where = f"turn {turn.index + 1} / {spec.label!r}"
        failed = {c.id for c in turn.report.failed_checks()}
        warned = {c.id for c in turn.report.warnings()}
        if spec.expect_pass != turn.passed:
            problems.append(
                f"{where}: expected {'PASS' if spec.expect_pass else 'FAIL'}, "
                f"got {turn.report.verdict.value} [{', '.join(sorted(failed))}]"
            )
        missing = set(spec.expect_fail) - failed
        if missing:
            problems.append(f"{where}: expected failing check(s) {sorted(missing)}")
        missing_warn = set(spec.expect_warn) - warned
        if missing_warn:
            problems.append(f"{where}: expected warning(s) {sorted(missing_warn)}")

    for text in scenario.expect_declared:
        if text not in outcome.declared:
            problems.append(f"declared limitations do not mention {text!r}")

    return problems


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def _indent(text: str, prefix: str = "    | ") -> str:
    return "\n".join(prefix + line for line in text.rstrip().splitlines())


def _print_transcript(scenario: Scenario, outcome: LoopOutcome) -> None:
    print(f"\nSCENARIO  {scenario.name}\n  {scenario.summary}")
    print(f"  budget {scenario.max_attempts} turns")
    for turn in outcome.turns:
        print(f"\n  TURN {turn.index + 1}  {turn.label}")
        if not turn.passed:
            print(_indent(turn.feedback))
    print(f"\n  -> {outcome.outcome} after {outcome.turns_used} turn(s)")
    if outcome.declared:
        print(_indent(outcome.declared))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rig.gate2_loop",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("scenarios", nargs="*", help="scenario names (default: all)")
    parser.add_argument("--list", action="store_true", help="list scenarios and exit")
    parser.add_argument("--workdir", help="where artifacts land (default: a temp dir)")
    parser.add_argument("--quiet", action="store_true", help="summary lines only")
    parser.add_argument("--json", action="store_true", help="emit JSON to stdout")
    args = parser.parse_args(argv)

    if args.list:
        for scenario in SCENARIOS.values():
            print(f"{scenario.name:<22} {scenario.summary}")
        return 0

    names = args.scenarios or list(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        parser.error(f"unknown scenario(s): {', '.join(unknown)}. Known: {', '.join(SCENARIOS)}")

    keep = args.workdir is not None
    root = Path(args.workdir).resolve() if keep else Path(tempfile.mkdtemp(prefix="gate2_loop_"))
    rows: list[tuple[Scenario, LoopOutcome, list[str]]] = []
    try:
        for name in names:
            scenario = SCENARIOS[name]
            # gated_review prints its verdict line; --json promises parseable stdout.
            with contextlib.redirect_stdout(io.StringIO()) if args.json else contextlib.nullcontext():
                outcome = run_gate2_loop(scenario, workdir=root / name)
            problems = check_expectations(scenario, outcome)
            rows.append((scenario, outcome, problems))
            if not (args.quiet or args.json):
                _print_transcript(scenario, outcome)
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)

    if args.json:
        print(json.dumps({"scenarios": [_as_dict(*row) for row in rows]}, indent=2))
    else:
        print()
        for scenario, outcome, problems in rows:
            status = "ok" if not problems else f"MISMATCH: {'; '.join(problems)}"
            print(f"  {scenario.name:<22} {outcome.outcome:<10} {outcome.turns_used} turn(s)  {status}")
    return 1 if any(problems for _, _, problems in rows) else 0


def _as_dict(scenario: Scenario, outcome: LoopOutcome, problems: list[str]) -> dict[str, Any]:
    return {
        "scenario": scenario.name,
        "outcome": outcome.outcome,
        "expected_outcome": scenario.expect_outcome,
        "turns_used": outcome.turns_used,
        "declared": outcome.declared,
        "problems": problems,
        "turns": [
            {
                "label": t.label,
                "verdict": t.report.verdict.value,
                "failed": sorted(c.id for c in t.report.failed_checks()),
                "warned": sorted(c.id for c in t.report.warnings()),
                "rejections_after": t.rejections_after,
            }
            for t in outcome.turns
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
