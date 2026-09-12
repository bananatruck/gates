"""Gate 2 tier A scored as a detector against a labelled corpus.

Every registry below carries a label and, when it is a defect, the key whose
defect the gate has to name. Running them produces a confusion matrix and the
rates derived from it, each with a Wilson interval and its denominator.

    python -m rig.gate2_tier_a_eval

**Read the positive rates with care.** Most of tier A is deterministic, so a
recall figure over defects we wrote is close to circular: it reports that checks
fire on the inputs they were written for. The honest reading is the split:

* The **negative** set is the informative half. Nothing guarantees a bound
  rejects only what it should, so the false-positive rate is a real measurement
  and it is the number that costs an engineer a revision for nothing. Its
  fixtures sit deliberately on the boundaries, at accuracy exactly 1.0, a loss
  of exactly 0.0, perplexity exactly 1.0, a speedup exactly at the ceiling.
* The **positive** set measures coverage rather than skill: whether a defect
  class has a check at all, and whether that check names the right key rather
  than failing for an unrelated reason.

Lives in ``rig/`` because ``pyproject.toml`` packages ``gates*`` only.
"""

from __future__ import annotations

import math
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from gates.gate2 import Gate2Config, Range, Relation, run_gate2
from gates.schema import Severity

DERIVES = Relation(key="a.speedup", op="ratio", left="a.slow_s", right="a.fast_s")
#: A relation whose right operand the run never recorded.
DANGLING = Relation(key="a.speedup", op="ratio", left="a.slow_s", right="a.never_recorded")


@dataclass(frozen=True)
class Case:
    """One registry, its label, and what the gate must do with it."""

    name: str
    values: dict[str, tuple[Any, str | None]]
    #: True when the registry contains a defect the gate must reject.
    defective: bool
    #: For a defect, the key the gate has to name. Failing on a different key is
    #: not catching this defect; it is a fixture hiding a broken check.
    subject: str = ""
    #: Which check should own it, for the per-check breakdown.
    expect_check: str = ""
    relations: tuple[Relation, ...] = ()
    ranges: dict[str, Range] = field(default_factory=dict)
    #: Why this case is in the corpus.
    note: str = ""


def _v(**kw: tuple[Any, str | None]) -> dict[str, tuple[Any, str | None]]:
    return dict(kw)


RANGE = "coherence.range_valid"
RELATION = "coherence.internal_consistency"
PLAUSIBLE = "coherence.plausibility"

# --------------------------------------------------------------------------- #
# positives: a defect the gate must reject, on the named key
# --------------------------------------------------------------------------- #

POSITIVES: tuple[Case, ...] = (
    Case("ratio_above_one", {"a.acc": (1.4, "ratio")}, True, "a.acc", RANGE),
    Case("ratio_below_zero", {"a.acc": (-0.1, "ratio")}, True, "a.acc", RANGE),
    Case("percent_above_hundred", {"a.p": (140.0, "percent")}, True, "a.p", RANGE),
    Case("f1_above_one", {"a.f1": (1.5, "f1")}, True, "a.f1", RANGE),
    Case("auc_below_zero", {"a.auc": (-0.02, "auc")}, True, "a.auc", RANGE),
    Case("precision_above_one", {"a.pr": (1.01, "precision")}, True, "a.pr", RANGE),
    Case("recall_above_one", {"a.rc": (1.2, "recall")}, True, "a.rc", RANGE),
    Case("perplexity_below_one", {"a.ppl": (0.3, "perplexity")}, True, "a.ppl", RANGE),
    Case("perplexity_just_below_one", {"a.ppl": (0.999, "perplexity")}, True, "a.ppl", RANGE,
         note="the boundary from the wrong side"),
    Case("negative_loss", {"a.loss": (-0.2, "loss")}, True, "a.loss", RANGE),
    Case("negative_count", {"a.n": (-5.0, "count")}, True, "a.n", RANGE),
    Case("zero_wallclock", {"a.t": (0.0, "seconds")}, True, "a.t", RANGE,
         note="an unmeasured run, not a fast one"),
    Case("negative_duration", {"a.t": (-1.0, "seconds")}, True, "a.t", RANGE),
    Case("nan_loss", {"a.loss": (float("nan"), "loss")}, True, "a.loss", RANGE),
    Case("inf_loss", {"a.loss": (float("inf"), "loss")}, True, "a.loss", RANGE),
    Case("nan_accuracy", {"a.acc": (float("nan"), "ratio")}, True, "a.acc", RANGE),
    Case("nan_wallclock", {"a.t": (float("nan"), "seconds")}, True, "a.t", RANGE),
    Case("nan_speedup", {"a.speedup": (float("nan"), "speedup")}, True, "a.speedup", RANGE),
    Case("inf_speedup", {"a.speedup": (float("inf"), "speedup")}, True, "a.speedup", RANGE),
    Case("nan_speedup_from_zero_divisor",
         _v(**{"a.slow_s": (13.61, "seconds"), "a.fast_s": (0.0, "seconds"),
               "a.speedup": (float("nan"), "speedup")}),
         True, "a.speedup", RANGE,
         note="the divisor is also a defect; the speedup must be named too"),
    Case("per_key_override_violated", {"a.acc": (0.80, "ratio")}, True, "a.acc", RANGE,
         ranges={"a.acc": Range(0.9, 1.0)},
         note="a declared narrower range beats the unit table"),
    Case("relation_does_not_hold",
         _v(**{"a.slow_s": (0.245, "seconds"), "a.fast_s": (0.018, "seconds"),
               "a.speedup": (2.0, "speedup")}),
         True, "a.speedup", RELATION, relations=(DERIVES,)),
    Case("relation_operand_never_recorded",
         _v(**{"a.slow_s": (0.245, "seconds"), "a.speedup": (13.6, "speedup")}),
         True, "a.speedup", RELATION, relations=(DANGLING,),
         note="declared but unfalsifiable, which must not read as holding"),
    Case("unexplained_speedup", {"a.speedup": (4000.0, "speedup")}, True, "a.speedup", PLAUSIBLE),
    Case("unexplained_speedup_just_over",
         {"a.speedup": (500.1, "speedup")}, True, "a.speedup", PLAUSIBLE,
         note="the ceiling from the wrong side"),
    Case("unexplained_speedup_absurd",
         {"a.speedup": (1e6, "speedup")}, True, "a.speedup", PLAUSIBLE),
    Case("relation_holds_but_speedup_still_unexplained",
         _v(**{"a.slow_s": (4000.0, "seconds"), "a.fast_s": (1.0, "seconds"),
               "a.speedup": (9999.0, "speedup")}),
         True, "a.speedup", RELATION, relations=(DERIVES,),
         note="a failing relation buys no exemption from the ceiling"),
)

# --------------------------------------------------------------------------- #
# negatives: a legitimate run the gate must not reject
#
# These are the informative half. Most sit exactly on a boundary, because that
# is where a bound written one comparison off rejects honest work.
# --------------------------------------------------------------------------- #

NEGATIVES: tuple[Case, ...] = (
    Case("clean_run",
         _v(**{"a.acc": (0.812, "ratio"), "a.slow_s": (0.245, "seconds"),
               "a.fast_s": (0.018, "seconds"), "a.speedup": (13.611, "speedup")}),
         False, relations=(DERIVES,)),
    Case("accuracy_exactly_one", {"a.acc": (1.0, "ratio")}, False,
         note="a perfect score is legal"),
    Case("accuracy_exactly_zero", {"a.acc": (0.0, "ratio")}, False),
    Case("loss_exactly_zero", {"a.loss": (0.0, "loss")}, False,
         note="loss >= 0 is closed; time > 0 is not"),
    Case("count_exactly_zero", {"a.n": (0.0, "count")}, False),
    Case("perplexity_exactly_one", {"a.ppl": (1.0, "perplexity")}, False,
         note="attainable: a model with zero cross entropy"),
    Case("percent_exactly_hundred", {"a.p": (100.0, "percent")}, False),
    Case("f1_at_both_bounds",
         _v(**{"a.f1": (1.0, "f1"), "a.pr": (0.0, "precision"), "a.rc": (1.0, "recall")}),
         False),
    Case("smallest_measurable_wallclock", {"a.t": (1e-9, "seconds")}, False,
         note="above the open bound, however narrowly"),
    Case("speedup_exactly_at_the_ceiling", {"a.speedup": (500.0, "speedup")}, False,
         note="the ceiling is exclusive; only above it is a defect"),
    Case("speedup_just_under_the_ceiling", {"a.speedup": (499.9, "speedup")}, False),
    Case("sage_derived_speedup",
         _v(**{"a.slow_s": (4700.0, "seconds"), "a.fast_s": (1.0, "seconds"),
               "a.speedup": (4700.0, "speedup")}),
         False, relations=(DERIVES,),
         note="SAGE arXiv 2606.31478 reports this ratio; rejecting it is a false positive"),
    Case("enormous_derived_speedup",
         _v(**{"a.slow_s": (1e9, "seconds"), "a.fast_s": (1.0, "seconds"),
               "a.speedup": (1e9, "speedup")}),
         False, relations=(DERIVES,),
         note="magnitude is not the test; provenance is"),
    Case("unit_the_gate_does_not_know", {"a.weird": (999.0, "furlongs")}, False,
         note="unchecked, and the report says so"),
    Case("no_unit_at_all", {"a.thing": (1.5, None)}, False),
    Case("non_numeric_values",
         _v(**{"a.note": ("done", "ratio"), "a.flag": (True, "ratio")}), False),
    Case("registry_with_no_values", {}, False,
         note="nothing to check is not the same as nothing wrong"),
    Case("large_clean_registry",
         {f"e{i}.acc": (0.8, "ratio") for i in range(50)}, False,
         note="50 legal values, none of which may trip anything"),
)

CASES = POSITIVES + NEGATIVES


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. The normal approximation is unusable at this n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _registry(case: Case) -> dict[str, Any]:
    return {
        "gate": "GATE 1 — EXECUTION VALIDITY",
        "verdict": "PASS",
        "citable": True,
        "values": {
            key: {"value": value, "unit": unit, "trace_id": f"t-{key}"}
            for key, (value, unit) in case.values.items()
        },
    }


def _flagged(check: Any) -> set[str]:
    keys: set[str] = set()
    for bucket in ("violations", "failures", "unresolved", "out_of_band"):
        for row in (check.evidence or {}).get(bucket) or []:
            if row.get("key"):
                keys.add(row["key"])
    return keys


def run(case: Case) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as artifacts:
        report = run_gate2(
            _registry(case),
            Gate2Config(
                artifact_root=artifacts,
                relations=case.relations,
                ranges=dict(case.ranges),
            ),
        )
    failed = [c for c in report.checks if not c.passed and c.severity is Severity.FAIL]
    naming = [c for c in failed if case.subject in _flagged(c)]

    if case.defective:
        outcome = "TP" if naming else "FN"
    else:
        outcome = "FP" if failed else "TN"

    return {
        "case": case.name,
        "outcome": outcome,
        "rejected": bool(failed),
        "by": naming[0].id if naming else (failed[0].id if failed else ""),
        "expected_check": case.expect_check,
        "right_check": bool(naming) and naming[0].id == case.expect_check,
    }


def main() -> int:
    rows = [run(c) for c in CASES]
    tally = Counter(r["outcome"] for r in rows)
    tp, fp, tn, fn = tally["TP"], tally["FP"], tally["TN"], tally["FN"]

    width = max(len(r["case"]) for r in rows)
    print("PER-CASE\n")
    print(f"{'case':{width}}  {'label':9}  {'outcome':7}  check")
    print("-" * (width + 46))
    for case, row in zip(CASES, rows):
        label = "defect" if case.defective else "legitimate"
        flag = "" if row["outcome"] in ("TP", "TN") else "   <-- WRONG"
        print(f"{row['case']:{width}}  {label:9}  {row['outcome']:7}  {row['by'] or '-'}{flag}")

    print("\n\nCONFUSION MATRIX\n")
    print(f"{'':22}{'gate rejected':>15}{'gate passed':>14}")
    print(f"{'registry has a defect':22}{tp:>15}{fn:>14}")
    print(f"{'registry is legitimate':22}{fp:>15}{tn:>14}")

    def rate(name: str, k: int, n: int, gloss: str) -> None:
        if n == 0:
            print(f"  {name:24} n/a   (denominator 0)")
            return
        lo, hi = wilson(k, n)
        print(f"  {name:24} {k/n:7.1%}   {k}/{n}   95% CI [{lo:.1%}, {hi:.1%}]   {gloss}")

    print("\n\nRATES, each with its denominator and a Wilson 95% interval\n")
    rate("detection rate", tp, tp + fn, "defects rejected on the right key")
    rate("false positive rate", fp, fp + tn, "legitimate runs wrongly rejected")
    rate("specificity", tn, tn + fp, "legitimate runs correctly passed")
    rate("precision", tp, tp + fp, "of everything rejected, truly defective")
    rate("accuracy", tp + tn, len(rows), "over the whole corpus")
    if tp:
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        print(f"  {'F1':24} {2 * prec * rec / (prec + rec):7.1%}")

    right = sum(1 for r in rows if r["expected_check"] and r["right_check"])
    labelled = sum(1 for c in CASES if c.expect_check)
    print()
    rate("check attribution", right, labelled, "caught by the check that should own it")

    wrong = [r for r in rows if r["outcome"] in ("FP", "FN")]
    print(f"\n{len(rows) - len(wrong)}/{len(rows)} cases classified as labelled")
    if wrong:
        print("MISCLASSIFIED: " + ", ".join(f"{r['case']} ({r['outcome']})" for r in wrong))
    return 1 if wrong else 0


if __name__ == "__main__":
    raise SystemExit(main())
