"""Gate 2 tier B scored against a labelled corpus.

Tier B carries two checks with two severities, so it is two detectors and gets
two confusion matrices:

* **``coherence.method_conformance``** (FAIL) must catch a run that used a value
  the plan did not declare, and must not fail a run that conformed.
* **``coherence.method_traceable``** (WARN) must flag a declared field nothing
  can be checked against, and must not flag one that can.

The third label is the one the design turns on. An *unverifiable* field is
neither conforming nor divergent, and scoring it as a divergence would be the
collapse this tier exists to avoid: "nobody can tell" is not "the run did
something else". Every unverifiable case is therefore scored twice, once as a
negative for the divergence detector and once as a positive for the traceability
detector, and a third rate reports how often the two were kept apart.

    python -m rig.gate2_tier_b_eval

Lives in ``rig/`` because ``pyproject.toml`` packages ``gates*`` only.
"""

from __future__ import annotations

import math
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from gates.gate2 import Gate2Config, PlanField, run_gate2

CONFORMANCE = "coherence.method_conformance"
TRACEABLE = "coherence.method_traceable"

DIVERGENT = "divergent"
CONFORMING = "conforming"
UNVERIFIABLE = "unverifiable"


@dataclass(frozen=True)
class Case:
    """One registry, one plan, and the outcome the tier must reach."""

    name: str
    #: key -> (value, unit, arg_kind). ``arg_kind`` of ``None`` omits provenance
    #: entirely, which is what a registry not built by Gate 1 looks like.
    values: dict[str, tuple[Any, str | None, str | None]]
    fields: tuple[PlanField, ...]
    label: str
    #: The key the tier has to name. Reaching the right verdict by naming some
    #: other field is not reaching it.
    subject: str
    note: str = ""


def f(key: str, declared: Any) -> PlanField:
    return PlanField(key=key, declared=declared, source_span=f"plan.md:{key}")


# --------------------------------------------------------------------------- #
# divergent: the run used something the plan did not declare
# --------------------------------------------------------------------------- #

DIVERGENCES: tuple[Case, ...] = (
    Case("float_differs", {"config.lr": (0.01, None, "constant")},
         (f("config.lr", 0.001),), DIVERGENT, "config.lr"),
    Case("integer_differs", {"config.batch": (64.0, "count", "constant")},
         (f("config.batch", 32),), DIVERGENT, "config.batch"),
    Case("string_differs", {"config.dataset": ("CIFAR-100", None, "constant")},
         (f("config.dataset", "CIFAR-10"),), DIVERGENT, "config.dataset"),
    Case("string_case_differs", {"config.dataset": ("cifar-10", None, "constant")},
         (f("config.dataset", "CIFAR-10"),), DIVERGENT, "config.dataset",
         note="case is not padding; CIFAR-10 and cifar-10 are not declared equal"),
    Case("boolean_differs", {"config.shuffle": (False, None, "constant")},
         (f("config.shuffle", True),), DIVERGENT, "config.shuffle"),
    Case("sign_flipped", {"config.margin": (0.5, None, "constant")},
         (f("config.margin", -0.5),), DIVERGENT, "config.margin"),
    Case("just_outside_representation_slack", {"config.x": (0.31, None, "constant")},
         (f("config.x", 0.1 + 0.2),), DIVERGENT, "config.x",
         note="0.3 conforms, 0.31 does not; the slack is representation only"),
    Case("zero_against_one", {"config.warmup": (1.0, "count", "constant")},
         (f("config.warmup", 0),), DIVERGENT, "config.warmup"),
    Case("number_recorded_as_string", {"config.lr": ("0.001", None, "constant")},
         (f("config.lr", 0.001),), DIVERGENT, "config.lr",
         note="a string is not the float it spells"),
    Case("order_of_magnitude", {"config.lr": (1.0, None, "constant")},
         (f("config.lr", 0.001),), DIVERGENT, "config.lr"),
    Case("one_of_several_diverges",
         {"config.lr": (0.001, None, "constant"),
          "config.batch": (32.0, "count", "constant"),
          "config.seed": (13.0, "count", "constant")},
         (f("config.lr", 0.001), f("config.batch", 32), f("config.seed", 7)),
         DIVERGENT, "config.seed",
         note="two conforming fields must not mask the third"),
    Case("divergent_and_untraceable_together",
         {"config.lr": (0.01, None, "constant")},
         (f("config.lr", 0.001), f("config.seed", 7)),
         DIVERGENT, "config.lr",
         note="a missing field alongside must not soften the divergence"),
)

# --------------------------------------------------------------------------- #
# conforming: the run did what the plan said
# --------------------------------------------------------------------------- #

CONFORMANCES: tuple[Case, ...] = (
    Case("exact_float", {"config.lr": (0.001, None, "constant")},
         (f("config.lr", 0.001),), CONFORMING, "config.lr"),
    Case("integer_declared_float_recorded", {"config.batch": (32.0, "count", "constant")},
         (f("config.batch", 32),), CONFORMING, "config.batch",
         note="32 and 32.0 are the same batch size"),
    Case("binary_representation", {"config.x": (0.3, None, "constant")},
         (f("config.x", 0.1 + 0.2),), CONFORMING, "config.x",
         note="0.1 + 0.2 is 0.30000000000000004 and that is not a divergence"),
    Case("scientific_notation", {"config.lr": (1e-3, None, "constant")},
         (f("config.lr", 0.001),), CONFORMING, "config.lr"),
    Case("string_with_padding", {"config.dataset": ("  CIFAR-10 ", None, "constant")},
         (f("config.dataset", "CIFAR-10"),), CONFORMING, "config.dataset"),
    Case("boolean_true", {"config.shuffle": (True, None, "constant")},
         (f("config.shuffle", True),), CONFORMING, "config.shuffle"),
    Case("zero_declared_zero_recorded", {"config.warmup": (0.0, "count", "constant")},
         (f("config.warmup", 0),), CONFORMING, "config.warmup"),
    Case("negative_value", {"config.margin": (-0.5, None, "constant")},
         (f("config.margin", -0.5),), CONFORMING, "config.margin"),
    Case("computed_value", {"config.steps": (1000.0, "count", "computed")},
         (f("config.steps", 1000),), CONFORMING, "config.steps",
         note="a value the run derived is stronger evidence than a constant"),
    Case("several_fields_all_conform",
         {"config.lr": (0.001, None, "constant"),
          "config.batch": (32.0, "count", "constant"),
          "config.dataset": ("CIFAR-10", None, "constant")},
         (f("config.lr", 0.001), f("config.batch", 32), f("config.dataset", "CIFAR-10")),
         CONFORMING, "config.lr"),
    Case("registry_holds_more_than_the_plan_declared",
         {"config.lr": (0.001, None, "constant"), "a.acc": (0.8, "ratio", "computed")},
         (f("config.lr", 0.001),), CONFORMING, "config.lr",
         note="undeclared values are not tier B's business"),
)

# --------------------------------------------------------------------------- #
# unverifiable: declared, but nothing can check it
#
# These must NOT reach coherence.method_conformance, and MUST reach
# coherence.method_traceable. Scoring one as a divergence is the collapse.
# --------------------------------------------------------------------------- #

UNVERIFIABLES: tuple[Case, ...] = (
    Case("never_recorded", {"a.acc": (0.8, "ratio", "computed")},
         (f("config.lr", 0.001),), UNVERIFIABLE, "config.lr",
         note="the plan claimed something the run never measured"),
    Case("recorded_as_a_call_site_literal", {"config.lr": (0.001, None, "literal")},
         (f("config.lr", 0.001),), UNVERIFIABLE, "config.lr",
         note="matching a typed number proves the agent typed it twice"),
    Case("literal_that_also_differs", {"config.lr": (0.01, None, "literal")},
         (f("config.lr", 0.001),), UNVERIFIABLE, "config.lr",
         note="untraceable outranks divergent: the recorded number is not evidence either way"),
    Case("no_provenance_at_all", {"config.lr": (0.001, None, None)},
         (f("config.lr", 0.001),), UNVERIFIABLE, "config.lr",
         note="a registry Gate 1 did not build; unknown is not fine"),
    Case("empty_registry", {}, (f("config.lr", 0.001),), UNVERIFIABLE, "config.lr"),
    Case("one_unverifiable_among_conforming",
         {"config.lr": (0.001, None, "constant"), "config.batch": (32.0, "count", "constant")},
         (f("config.lr", 0.001), f("config.batch", 32), f("config.seed", 7)),
         UNVERIFIABLE, "config.seed",
         note="two checkable fields must not hide the third"),
)

CASES = DIVERGENCES + CONFORMANCES + UNVERIFIABLES


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _registry(case: Case) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, (value, unit, arg_kind) in case.values.items():
        entry: dict[str, Any] = {"value": value, "unit": unit, "trace_id": f"t-{key}"}
        if arg_kind is not None:
            entry["provenance"] = {"arg_kind": arg_kind}
        values[key] = entry
    return {"gate": "GATE 1 — EXECUTION VALIDITY", "verdict": "PASS",
            "citable": True, "values": values}


def run(case: Case) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as artifacts:
        report = run_gate2(
            _registry(case),
            Gate2Config(artifact_root=artifacts, plan_fields=case.fields),
        )
    by_id = {c.id: c for c in report.checks}
    conf, trace = by_id.get(CONFORMANCE), by_id.get(TRACEABLE)

    named_divergent = bool(conf) and case.subject in {
        row["key"] for row in conf.evidence.get("divergent", [])
    }
    named_untraceable = bool(trace) and case.subject in {
        row["key"] for row in trace.evidence.get("unverifiable", [])
    }
    named_conforming = bool(conf) and case.subject in conf.evidence.get("conforming", [])

    if case.label == DIVERGENT:
        divergence = "TP" if named_divergent else "FN"
        traceability = "FP" if named_untraceable else "TN"
        correct = named_divergent and not named_untraceable
    elif case.label == CONFORMING:
        divergence = "FP" if named_divergent else "TN"
        traceability = "FP" if named_untraceable else "TN"
        correct = named_conforming and not named_divergent and not named_untraceable
    else:
        # The anti-collapse case: must not be a divergence, must be untraceable.
        divergence = "FP" if named_divergent else "TN"
        traceability = "TP" if named_untraceable else "FN"
        correct = named_untraceable and not named_divergent

    return {
        "case": case.name,
        "label": case.label,
        "divergence": divergence,
        "traceability": traceability,
        "correct": correct,
        "verdict": report.verdict.value,
    }


def _matrix(title: str, rows: list[dict], key: str, pos: str, neg: str) -> Counter:
    tally = Counter(r[key] for r in rows)
    print(f"\n{title}\n")
    print(f"{'':36}{'flagged':>10}{'not flagged':>14}")
    print(f"{pos:36}{tally['TP']:>10}{tally['FN']:>14}")
    print(f"{neg:36}{tally['FP']:>10}{tally['TN']:>14}")
    return tally


def _rate(name: str, k: int, n: int, gloss: str) -> None:
    if n == 0:
        print(f"  {name:26} n/a  (denominator 0)")
        return
    lo, hi = wilson(k, n)
    print(f"  {name:26} {k/n:7.1%}   {k}/{n}   95% CI [{lo:.1%}, {hi:.1%}]   {gloss}")


def main() -> int:
    rows = [run(c) for c in CASES]
    width = max(len(r["case"]) for r in rows)

    print("PER-CASE\n")
    print(f"{'case':{width}}  {'label':13}  {'divergence':11}  {'traceability':12}  ok")
    print("-" * (width + 46))
    for row in rows:
        print(f"{row['case']:{width}}  {row['label']:13}  {row['divergence']:11}  "
              f"{row['traceability']:12}  {'y' if row['correct'] else 'N'}")

    d = _matrix("DIVERGENCE DETECTOR  ·  coherence.method_conformance (FAIL)",
                rows, "divergence", "run used another value", "conformed, or nothing to check")
    t = _matrix("TRACEABILITY DETECTOR  ·  coherence.method_traceable (WARN)",
                rows, "traceability", "field cannot be checked", "field can be checked")

    print("\n\nRATES, each with its denominator and a Wilson 95% interval\n")
    _rate("divergence detection", d["TP"], d["TP"] + d["FN"], "real divergences caught")
    _rate("divergence FPR", d["FP"], d["FP"] + d["TN"], "conforming or unverifiable called divergent")
    _rate("traceability detection", t["TP"], t["TP"] + t["FN"], "unverifiable fields flagged")
    _rate("traceability FPR", t["FP"], t["FP"] + t["TN"], "checkable fields called unverifiable")

    unver = [r for r in rows if r["label"] == UNVERIFIABLE]
    kept = sum(1 for r in unver if r["divergence"] == "TN" and r["traceability"] == "TP")
    print()
    _rate("outcomes kept apart", kept, len(unver),
          "unverifiable reported as itself, never as a divergence")

    correct = sum(1 for r in rows if r["correct"])
    _rate("overall", correct, len(rows), "case reached its labelled outcome")

    wrong = [r for r in rows if not r["correct"]]
    print(f"\n{correct}/{len(rows)} cases reached the labelled outcome")
    if wrong:
        print("MISCLASSIFIED: " + ", ".join(f"{r['case']} ({r['label']})" for r in wrong))
    return 1 if wrong else 0


if __name__ == "__main__":
    raise SystemExit(main())
