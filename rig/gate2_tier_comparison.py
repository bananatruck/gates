"""Gate 2's three tiers compared: what each one adds over the tiers before it.

    python -m rig.gate2_tier_comparison

**A against A+B.** Every case in the two labelled corpora the tier evaluations
already score (`rig/gate2_tier_a_eval.py`, `rig/gate2_tier_b_eval.py`) runs twice:
once with the plan withheld, which is tier A alone, and once with it, which is
A+B. There is no B-alone column. Ranges run on every registry, so tier A cannot
be switched off, and a column that pretended otherwise would report a
configuration no host can build.

**A+B against A+B+C.** Tier C adds no check, so it is compared on what happens
after detection, over the loop scenarios (`rig/gate2_scenarios.py`). One shot is
the first review and nothing more: whatever it found stands declared. The loop
sends the finding back, runs the fix under Gate 1, and reviews again.

Lives in ``rig/`` because ``pyproject.toml`` packages ``gates*`` only.
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gates.gate2 import Gate2Config, run_gate2  # noqa: E402
from gates.schema import Severity  # noqa: E402

from rig import gate2_tier_a_eval as tier_a  # noqa: E402
from rig import gate2_tier_b_eval as tier_b  # noqa: E402
from rig.gate2_loop import run_gate2_loop  # noqa: E402
from rig.gate2_scenarios import SCENARIOS  # noqa: E402


def _flagged(case: tier_b.Case, *, plan: bool, severities=(Severity.FAIL, Severity.WARN)) -> bool:
    """Did any check at these severities fire on a tier B case, with or without its plan?"""
    fields = case.fields if plan else ()
    with tempfile.TemporaryDirectory() as artifacts:
        report = run_gate2(
            tier_b._registry(case), Gate2Config(artifact_root=artifacts, plan_fields=fields)
        )
    return any(not c.passed and c.severity in severities for c in report.checks)


def _rejected(case: tier_b.Case, *, plan: bool) -> bool:
    return _flagged(case, plan=plan, severities=(Severity.FAIL,))


def detection() -> dict[str, dict[str, int]]:
    """Caught and wrongly rejected, per defect class, for A and for A+B."""
    a_rows = [tier_a.run(c) for c in tier_a.CASES]
    b_rows = [(c, tier_b.run(c)) for c in tier_b.CASES]
    by_label = lambda label: [(c, r) for c, r in b_rows if c.label == label]  # noqa: E731

    boundary = sum(r["outcome"] == "TP" for r in a_rows)
    divergent, unverifiable = by_label(tier_b.DIVERGENT), by_label(tier_b.UNVERIFIABLE)
    legit_b = by_label(tier_b.CONFORMING) + unverifiable
    return {
        # Tier A cases carry no plan, so A and A+B are the same run.
        "boundary_defects": {
            "n": sum(c.defective for c in tier_a.CASES), "A": boundary, "A+B": boundary,
        },
        "plan_divergences": {
            "n": len(divergent),
            "A": sum(_flagged(c, plan=False) for c, _ in divergent),
            "A+B": sum(r["divergence"] == "TP" for _, r in divergent),
        },
        "unverifiable_fields": {
            "n": len(unverifiable),
            "A": sum(_flagged(c, plan=False) for c, _ in unverifiable),
            "A+B": sum(r["traceability"] == "TP" for _, r in unverifiable),
        },
        # An unverifiable field must warn, never fail, so it counts as legitimate
        # here: a FAIL on it is a false rejection.
        "false_rejections": {
            "n": sum(not c.defective for c in tier_a.CASES) + len(legit_b),
            "A": sum(r["outcome"] == "FP" for r in a_rows)
            + sum(_rejected(c, plan=False) for c, _ in legit_b),
            "A+B": sum(r["outcome"] == "FP" for r in a_rows)
            + sum(_rejected(c, plan=True) for c, _ in legit_b),
        },
    }


def loop() -> dict[str, int]:
    """One-shot review against the loop, over every scenario."""
    with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
        outcomes = [run_gate2_loop(s, workdir=Path(root) / s.name) for s in SCENARIOS.values()]
    reviewed = [[t for t in o.turns if t.gate == 2] for o in outcomes]
    entering = [o for o, turns in zip(outcomes, reviewed, strict=True) if not turns[0].passed]
    return {
        "runs": len(outcomes),
        "entering": len(entering),
        # One shot: the first review is final, so nothing it found gets fixed.
        "fixed_one_shot": 0,
        "fixed_with_loop": sum(o.outcome == "pass" for o in entering),
        "declared_one_shot": len(entering),
        "declared_with_loop": sum(o.outcome != "pass" for o in entering),
        # A fix only ever reaches Gate 2 through Gate 1 (F12).
        "fixes_gate1_rejected": sum(t.gate == 1 for o in outcomes for t in o.turns),
    }


def main() -> int:
    table = detection()
    print("TIERS A AND A+B, over the tier A and tier B corpora\n")
    print(f"{'':22}{'n':>5}{'A':>8}{'A+B':>8}")
    for name, row in table.items():
        print(f"{name:22}{row['n']:>5}{row['A']:>8}{row['A+B']:>8}")

    runs = loop()
    print(f"\nTIER C, over {runs['runs']} loop scenarios, "
          f"{runs['entering']} with a finding on first review\n")
    print(f"{'':22}{'A+B':>8}{'A+B+C':>8}")
    print(f"{'fixed before writing':22}{runs['fixed_one_shot']:>8}{runs['fixed_with_loop']:>8}")
    print(f"{'declared to the writer':22}{runs['declared_one_shot']:>8}"
          f"{runs['declared_with_loop']:>8}")
    print(f"\n{runs['fixes_gate1_rejected']} fix(es) rejected by Gate 1 before Gate 2 saw them")

    missed = any(table[k]["A+B"] < table[k]["n"]
                 for k in ("boundary_defects", "plan_divergences", "unverifiable_fields"))
    wrong = table["false_rejections"]["A+B"] > 0
    return 1 if missed or wrong else 0


if __name__ == "__main__":
    raise SystemExit(main())
