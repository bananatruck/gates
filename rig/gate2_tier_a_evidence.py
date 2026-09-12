"""Gate 2 tier A, one fixture per defect class, run and tabulated.

Evidence rather than assertion. Every row below is produced by calling
``run_gate2`` on a registry built here, so the table in
``docs/research/gate2-tier-a-evidence.md`` is reproducible by running this file
rather than by trusting it.

Lives in ``rig/`` because ``pyproject.toml`` packages ``gates*`` only, so nothing
here can reach a scaffold that installs the library.

    python -m rig.gate2_tier_a_evidence

Point it at an older checkout to measure what tier A changed:

    git worktree add .cache/before <sha>
    cp rig/gate2_tier_a_evidence.py .cache/before/rig/
    (cd .cache/before && python -m rig.gate2_tier_a_evidence)
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Any

from gates.gate2 import Gate2Config, Relation, run_gate2
from gates.schema import Severity

#: The relation that derives a speedup from the two times it was measured
#: against. Declaring it is what buys the exemption in ``sage_derived_speedup``.
DERIVES = Relation(key="a.speedup", op="ratio", left="a.slow_s", right="a.fast_s")


@dataclass(frozen=True)
class Fixture:
    """One defect class, and what the gate is supposed to do about it."""

    name: str
    #: Why this row exists. Printed in the document, not in the table.
    defect: str
    values: dict[str, tuple[Any, str | None]]
    relations: tuple[Relation, ...] = ()
    #: True when the fixture is a legitimate run the gate must not reject.
    must_pass: bool = False
    #: The key whose defect must be named. A report that fails for a *different*
    #: reason is a fixture hiding a broken check, so the verdict alone is not
    #: enough evidence: the gate has to flag the value this fixture is about.
    subject: str = ""


FIXTURES: tuple[Fixture, ...] = (
    Fixture(
        "clean_run",
        "a run with nothing wrong with it",
        {"a.acc": (0.812, "ratio"), "a.slow_s": (0.245, "seconds"),
         "a.fast_s": (0.018, "seconds"), "a.speedup": (13.611, "speedup")},
        relations=(DERIVES,),
        must_pass=True,
    ),
    Fixture(
        "impossible_accuracy",
        "a ratio above 1.0",
        {"a.acc": (1.4, "ratio")},
        subject="a.acc",
    ),
    Fixture(
        "sub_unit_perplexity",
        "perplexity below 1, which exp(H) cannot produce for H >= 0",
        {"a.ppl": (0.3, "perplexity")},
        subject="a.ppl",
    ),
    Fixture(
        "f1_above_one",
        "a bounded score outside its definition",
        {"a.f1": (1.5, "f1")},
        subject="a.f1",
    ),
    Fixture(
        "negative_loss",
        "a loss below zero",
        {"a.loss": (-0.2, "loss")},
        subject="a.loss",
    ),
    Fixture(
        "unmeasured_wallclock",
        "a duration of exactly zero, which is an unmeasured run not a fast one",
        {"a.t": (0.0, "seconds")},
        subject="a.t",
    ),
    Fixture(
        "nan_speedup_from_zero_divisor",
        "the speedup derived from that unmeasured wallclock",
        {"a.slow_s": (13.61, "seconds"), "a.fast_s": (0.0, "seconds"),
         "a.speedup": (float("nan"), "speedup")},
        subject="a.speedup",
    ),
    Fixture(
        "infinite_loss",
        "a diverged loss reported as inf",
        {"a.loss": (float("inf"), "loss")},
        subject="a.loss",
    ),
    Fixture(
        "broken_relation",
        "a speedup that contradicts the two times it is declared to derive from",
        {"a.slow_s": (0.245, "seconds"), "a.fast_s": (0.018, "seconds"),
         "a.speedup": (2.0, "speedup")},
        relations=(DERIVES,),
        subject="a.speedup",
    ),
    Fixture(
        "unexplained_speedup",
        "4000x with nothing deriving it",
        {"a.speedup": (4000.0, "speedup")},
        subject="a.speedup",
    ),
    Fixture(
        "sage_derived_speedup",
        "SAGE's real 4,700x FVA-over-FBA ratio, derived from two recorded times",
        {"a.slow_s": (4700.0, "seconds"), "a.fast_s": (1.0, "seconds"),
         "a.speedup": (4700.0, "speedup")},
        relations=(DERIVES,),
        must_pass=True,
    ),
    Fixture(
        "unknown_unit",
        "a value this gate has no range for, which must read as unchecked",
        {"a.weird": (999.0, "furlongs")},
        must_pass=True,
    ),
)


def _registry(fixture: Fixture) -> dict[str, Any]:
    return {
        "gate": "GATE 1 — EXECUTION VALIDITY",
        "verdict": "PASS",
        "citable": True,
        "values": {
            key: {"value": value, "unit": unit, "trace_id": f"t-{key}"}
            for key, (value, unit) in fixture.values.items()
        },
    }


def _flagged_keys(check: Any) -> set[str]:
    """Every registry key one failed check names, whatever its evidence shape."""
    evidence = check.evidence or {}
    keys: set[str] = set()
    for bucket in ("violations", "failures", "unresolved", "out_of_band"):
        for row in evidence.get(bucket) or []:
            key = row.get("key")
            if key:
                keys.add(key)
    return keys


def run(fixture: Fixture) -> dict[str, Any]:
    """Run one fixture and reduce the report to the row the table prints."""
    with tempfile.TemporaryDirectory() as artifacts:
        report = run_gate2(
            _registry(fixture),
            Gate2Config(artifact_root=artifacts, relations=fixture.relations),
        )
    failed = [c for c in report.checks if not c.passed and c.severity is Severity.FAIL]
    naming = [c for c in failed if fixture.subject in _flagged_keys(c)]

    if fixture.must_pass:
        correct = report.passed
    else:
        # Rejecting for some other reason is not catching this defect.
        correct = bool(naming)

    return {
        "fixture": fixture.name,
        "verdict": report.verdict.value,
        "caught_by": naming[0].id if naming else (failed[0].id + " (other key)" if failed else ""),
        "emitted": [c.id for c in report.checks],
        "correct": correct,
    }


def main() -> int:
    rows = [run(f) for f in FIXTURES]
    width = max(len(r["fixture"]) for r in rows)

    print(f"{'fixture':{width}}  {'expect':7}  {'verdict':7}  {'caught by':30}  ok")
    print("-" * (width + 56))
    for fixture, row in zip(FIXTURES, rows):
        expect = "pass" if fixture.must_pass else "reject"
        print(
            f"{row['fixture']:{width}}  {expect:7}  {row['verdict']:7}  "
            f"{row['caught_by'] or '-':30}  {'y' if row['correct'] else 'N'}"
        )

    wrong = [r for r in rows if not r["correct"]]
    print()
    print(f"{len(rows) - len(wrong)}/{len(rows)} fixtures behaved as specified")
    if wrong:
        print("MISBEHAVED: " + ", ".join(r["fixture"] for r in wrong))
    return 1 if wrong else 0


if __name__ == "__main__":
    raise SystemExit(main())
