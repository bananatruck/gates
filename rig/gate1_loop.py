"""Run the Gate 1 feedback loop and show it working.

    python -m rig.gate1_loop                  # every scenario
    python -m rig.gate1_loop archived-run     # one
    python -m rig.gate1_loop --list
    python -m rig.gate1_loop --quiet --json   # for CI

Exits non-zero if any scenario departs from what it documents, so this is a test
as much as a demonstration. No API key, no model, no training run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gates import render_summary  # noqa: E402

from rig.loop import (  # noqa: E402
    ExecutionOutcome,
    LoopOutcome,
    check_expectations,
    run_loop,
)
from rig.scenarios import SCENARIOS, Scenario  # noqa: E402

RULE = "═" * 78
THIN = "─" * 78


# --------------------------------------------------------------------------- #
# transcript
# --------------------------------------------------------------------------- #


class Transcript:
    """Streams the loop as it happens, so a long run is not a silent one."""

    def __init__(self, *, quiet: bool = False, show_feedback: bool = True):
        self.quiet = quiet
        self.show_feedback = show_feedback

    def __call__(self, event: str, payload: dict[str, Any]) -> None:
        if self.quiet:
            return
        handler = getattr(self, f"_on_{event}", None)
        if handler:
            handler(payload)

    def _on_loop_start(self, payload: dict[str, Any]) -> None:
        scenario: Scenario = payload["scenario"]
        keys = ", ".join(scenario.expected_keys or ()) or "(any, at least one)"
        print(f"\n{RULE}")
        print(f"SCENARIO  {scenario.name}")
        print(f"  {scenario.summary}")
        print(f"  budget {scenario.max_attempts} rewrites  ·  declared keys: {keys}")
        print(RULE)

    def _on_turn_start(self, payload: dict[str, Any]) -> None:
        turn = payload["turn"]
        steps = payload["steps"]
        extra = f"  ({len(steps)} executions — inner repair loop)" if len(steps) > 1 else ""
        print(f"\n{THIN}")
        print(f"TURN {turn.index + 1}{extra}")

    def _on_execution(self, payload: dict[str, Any]) -> None:
        outcome: ExecutionOutcome = payload["outcome"]
        print(f"\n  ▸ {outcome.label}")
        print(f"    {render_summary(outcome.report)}")
        print(f"    {_upstream_line(outcome)}")
        if self.show_feedback:
            body = (
                outcome.feedback
                if not outcome.passed
                else payload["gated"].evidence_bundle
            )
            print()
            print(_indent(body, "    | "))

    def _on_turn_end(self, payload: dict[str, Any]) -> None:
        turn = payload["turn"]
        context = payload["context"]
        verdict = "PASS" if turn.passed else "FAIL"
        print(
            f"\n  turn {turn.index + 1} closed: {verdict}  ·  budget "
            f"{turn.rejections_after}/{context.config.max_attempts} consecutive "
            f"rejections  ·  {context.attempt} execution(s) so far"
        )

    def _on_gate_failure(self, payload: dict[str, Any]) -> None:
        print(f"\n  GateFailure raised — no paper is produced.")
        print(_indent(str(payload["error"]), "    | "))


def _upstream_line(outcome: ExecutionOutcome) -> str:
    """What the 1000-character channel would have delivered for this attempt."""
    if outcome.passed:
        return "upstream: n/a — the run was admitted"
    if outcome.upstream_detected_failure is None:
        return "upstream: not reconstructed"
    reconstructed = " (reconstructed by executing it)" if outcome.counterfactual_executed else ""
    if outcome.upstream_detected_failure:
        return f"upstream: would have caught this too{reconstructed}"
    if outcome.marker_visible_at is None:
        return (
            "upstream: would have ACCEPTED this run — it exited 0 and printed "
            f"numbers, so no marker existed to find{reconstructed}"
        )
    return (
        "upstream: would have ACCEPTED this run — crash marker lands at "
        f"character {outcome.marker_visible_at:,}, past the 1,000-character "
        f"ceiling{reconstructed}"
    )


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.rstrip().splitlines())


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #


def summarise(rows: list[tuple[Scenario, LoopOutcome, list[str]]]) -> str:
    out = [f"\n{RULE}", "SUMMARY", RULE, ""]
    header = f"  {'SCENARIO':<16} {'TURNS':>5} {'EXEC':>5}  {'OUTCOME':<13} {'UPSTREAM BLIND':<15} RESULT"
    out.append(header)
    out.append("  " + "-" * (len(header) - 2))
    for scenario, outcome, problems in rows:
        blind = outcome.upstream_blind_turns
        blind_text = (
            f"{len(blind)}/{outcome.turns_used} turns" if blind else "—"
        )
        status = "ok" if not problems else f"MISMATCH ({len(problems)})"
        out.append(
            f"  {scenario.name:<16} {outcome.turns_used:>5} "
            f"{len(outcome.executions):>5}  {outcome.outcome:<13} "
            f"{blind_text:<15} {status}"
        )

    failures = [(s, p) for s, _, p in rows if p]
    if failures:
        out.append("")
        out.append("  MISMATCHES")
        for scenario, problems in failures:
            for problem in problems:
                out.append(f"    [{scenario.name}] {problem}")
    else:
        out.append("")
        out.append("  Every scenario behaved exactly as documented.")

    total_blind = sum(len(o.upstream_blind_turns) for _, o, _ in rows)
    total_rejected = sum(
        1 for _, o, _ in rows for t in o.turns if not t.passed
    )
    out.append("")
    if all(o.counterfactual for _, o, _ in rows):
        out.append(
            f"  Gate 1 rejected {total_rejected} engineer turn(s); upstream's "
            f"1,000-character detector would have accepted {total_blind} of them."
        )
    else:
        out.append(
            f"  Gate 1 rejected {total_rejected} engineer turn(s). The upstream "
            f"channel was not reconstructed, so how many it would have accepted "
            f"is unmeasured — not zero."
        )
    out.append("")
    return "\n".join(out)


def as_json(rows: list[tuple[Scenario, LoopOutcome, list[str]]]) -> str:
    payload = []
    for scenario, outcome, problems in rows:
        payload.append(
            {
                "scenario": scenario.name,
                "outcome": outcome.outcome,
                "expected_outcome": scenario.expect_outcome,
                "turns_used": outcome.turns_used,
                "executions": len(outcome.executions),
                "upstream_blind_turns": list(outcome.upstream_blind_turns),
                "fell_back": outcome.fell_back,
                "ledger": outcome.ledger_path,
                "artifacts": outcome.artifact_root,
                "problems": problems,
                "turns": [
                    {
                        "index": t.index,
                        "passed": t.passed,
                        "rejections_after": t.rejections_after,
                        "executions": [
                            {
                                "label": e.label,
                                "attempt": e.attempt,
                                "verdict": e.report.verdict.value,
                                "failed": sorted(e.failed_check_ids),
                                "warned": sorted(e.warned_check_ids),
                                "upstream_detected_failure": (
                                    e.upstream_detected_failure
                                ),
                                "marker_visible_at": e.marker_visible_at,
                            }
                            for e in t.executions
                        ],
                    }
                    for t in outcome.turns
                ],
            }
        )
    return json.dumps({"scenarios": payload}, indent=2)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rig.gate1_loop",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "scenarios",
        nargs="*",
        help="scenario names to run (default: all)",
    )
    parser.add_argument("--list", action="store_true", help="list scenarios and exit")
    parser.add_argument(
        "--workdir",
        help="where artifacts land (default: a temp dir, removed afterwards)",
    )
    parser.add_argument("--quiet", action="store_true", help="summary table only")
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="omit the feedback report bodies from the transcript",
    )
    parser.add_argument(
        "--no-counterfactual",
        action="store_true",
        help="skip reconstructing what upstream's channel would have delivered",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON to stdout")
    parser.add_argument(
        "--timeout", type=int, default=60, help="per-execution timeout (seconds)"
    )
    args = parser.parse_args(argv)

    if args.list:
        for scenario in SCENARIOS.values():
            print(f"{scenario.name:<16} {scenario.summary}")
        return 0

    names = args.scenarios or list(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        parser.error(
            f"unknown scenario(s): {', '.join(unknown)}. "
            f"Known: {', '.join(SCENARIOS)}"
        )

    keep = args.workdir is not None
    root = Path(args.workdir) if keep else Path(tempfile.mkdtemp(prefix="gate1_loop_"))
    quiet = args.quiet or args.json

    rows: list[tuple[Scenario, LoopOutcome, list[str]]] = []
    try:
        for name in names:
            scenario = SCENARIOS[name]
            outcome = run_loop(
                scenario,
                workdir=root / name,
                counterfactual=not args.no_counterfactual,
                timeout_s=args.timeout,
                observer=Transcript(
                    quiet=quiet, show_feedback=not args.no_feedback
                ),
            )
            rows.append((scenario, outcome, check_expectations(scenario, outcome)))

        if args.json:
            print(as_json(rows))
        else:
            print(summarise(rows))
            if keep:
                print(f"  Artifacts kept in {root}\n")
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)

    return 1 if any(problems for _, _, problems in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
