"""Run Gate 3's feedback loop and show it working.

    python -m rig.gate3_loop                     # every scenario
    python -m rig.gate3_loop budget-exhausts     # one
    python -m rig.gate3_loop --list
    python -m rig.gate3_loop --quiet --json      # for CI

Gate 3's loop is ``report_loop`` in the adapter, the same shape as Gate 2's
``review_loop`` (D29): the writer submits a manuscript, the gate issues a
verdict, a rejection renders a feedback report and costs one agent turn, and the
budget lives in the adapter's ``GateContext``, never in the gate.

The registry every manuscript is judged against comes from playing a Gate 2
scenario, ``clean`` unless the scenario names another, so it was written by a
run Gate 1 executed and Gate 2 reviewed. A hand-built registry would check a
table no run wrote (F12). Gate 2's declared limitations come with it.

One deliberate difference from Gate 2, `CLAUDE.md` §4. A spent budget raises
``GateFailure`` and no manuscript is emitted.

Exits non-zero if any scenario departs from what it documents. No model, no API
key.
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
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gates import GateFailure, GateReport, render_feedback  # noqa: E402
from gates.adapters.agentlab import make_report_context, report_loop  # noqa: E402

from rig.gate2_loop import run_gate2_loop  # noqa: E402
from rig.gate2_scenarios import SCENARIOS as GATE2_SCENARIOS  # noqa: E402
from rig.gate3_scenarios import SCENARIOS, Scenario, Turn  # noqa: E402


class ScriptedWriter:
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

    @property
    def passed(self) -> bool:
        return self.report.passed


@dataclass
class LoopOutcome:
    scenario: str
    turns: list[TurnOutcome] = field(default_factory=list)
    #: "pass" - a manuscript was admitted. "raised" - the budget was spent and
    #: ``GateFailure`` propagated. "no_pass" - the writer stopped first.
    outcome: str = "no_pass"
    #: The admitted manuscript, rendered. ``None`` unless Gate 3 passed it.
    manuscript: str | None = None
    #: What the manuscripts were judged against, from the scenario's Gate 2 run.
    registry: dict[str, Any] | None = None
    #: What that Gate 2 run declared, which the manuscript must state.
    declared: str = ""
    ledger_path: str | None = None

    @property
    def turns_used(self) -> int:
        return len(self.turns)


def run_gate3_loop(
    scenario: Scenario,
    *,
    workdir: str | Path,
    writer: ScriptedWriter | None = None,
) -> LoopOutcome:
    """Play one scenario through the real Gates 1, 2 and 3 and return what happened."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    writer = writer or ScriptedWriter(scenario)

    reviewed = run_gate2_loop(GATE2_SCENARIOS[scenario.gate2], workdir=workdir)
    context = make_report_context(research_dir=str(workdir), max_attempts=scenario.max_attempts)
    outcome = LoopOutcome(
        scenario=scenario.name,
        registry=reviewed.registry,
        declared=reviewed.declared,
        ledger_path=str(context.ledger.path) if context.ledger else None,
    )

    submitted: list[Turn] = []

    def write(feedback: str | None) -> str | None:
        turn = writer.turn(feedback, len(submitted))
        if turn is None:
            return None
        submitted.append(turn)
        return turn.manuscript

    try:
        written = report_loop(
            context,
            write,
            registry=reviewed.registry,
            declared=reviewed.declared,
            extra={"scenario": scenario.name},
        )
        outcome.outcome = written.outcome
        outcome.manuscript = written.manuscript
    except GateFailure:
        # report_loop never returns on a spent budget, and the exception carries
        # only the last report. The context kept all of them.
        outcome.outcome = "raised"

    for index, (turn, report) in enumerate(zip(submitted, context.history)):
        # The same text report_loop sent the writer: render_feedback is a pure
        # function of the report.
        outcome.turns.append(TurnOutcome(index, turn.label, report, render_feedback(report)))

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
    if (outcome.manuscript is not None) != (outcome.outcome == "pass"):
        problems.append(
            f"outcome {outcome.outcome!r} came with "
            f"{'a' if outcome.manuscript is not None else 'no'} manuscript"
        )

    for turn, spec in zip(outcome.turns, scenario.turns):
        where = f"turn {turn.index + 1} / {spec.label!r}"
        failed = {c.id for c in turn.report.failed_checks()}
        if spec.expect_pass != turn.passed:
            problems.append(
                f"{where}: expected {'PASS' if spec.expect_pass else 'FAIL'}, "
                f"got {turn.report.verdict.value} [{', '.join(sorted(failed))}]"
            )
        missing = set(spec.expect_fail) - failed
        if missing:
            problems.append(f"{where}: expected failing check(s) {sorted(missing)}")
        for text in spec.expect_feedback:
            if text not in turn.feedback:
                problems.append(f"{where}: feedback does not mention {text!r}")

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
    if outcome.manuscript is not None:
        print(_indent(outcome.manuscript))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rig.gate3_loop",
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
    root = Path(args.workdir).resolve() if keep else Path(tempfile.mkdtemp(prefix="gate3_loop_"))
    rows: list[tuple[Scenario, LoopOutcome, list[str]]] = []
    try:
        for name in names:
            scenario = SCENARIOS[name]
            # The gates print verdict lines; --json promises parseable stdout.
            with contextlib.redirect_stdout(io.StringIO()) if args.json else contextlib.nullcontext():
                outcome = run_gate3_loop(scenario, workdir=root / name)
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
        "manuscript_emitted": outcome.manuscript is not None,
        "problems": problems,
        "turns": [
            {
                "label": t.label,
                "verdict": t.report.verdict.value,
                "failed": sorted(c.id for c in t.report.failed_checks()),
                "warned": sorted(c.id for c in t.report.warnings()),
            }
            for t in outcome.turns
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
