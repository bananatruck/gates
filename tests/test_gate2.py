"""Gate 2's deterministic tier, checked against registries built by hand.

By hand rather than by running Gate 1: Gate 2's contract is the registry
*format*, and a test that has to execute an experiment to reach Gate 2 tests
Gate 1 twice.
"""

from __future__ import annotations

import json
import pathlib

import pytest

import gates.gate2_semantic
from gates.adapters.agentlab import (
    GateContext,
    declared_limitations,
    gated_review,
    make_review_context,
)
from gates.errors import GateError, GateFailure
from gates.gate2 import (
    GATE_NAME,
    IMPLAUSIBLE_SPEEDUP,
    Band,
    Gate2Config,
    Range,
    Relation,
    SourceClaim,
    band_for,
    run_gate2,
    unresolved_discrepancies,
)
from gates.report import render_evidence, render_feedback
from gates.schema import Severity, Verdict


def registry(values: dict[str, tuple], *, citable: bool = True) -> dict:
    """A minimal registry of the shape ``gates.registry.build_registry`` emits."""
    return {
        "gate": "GATE 1 — EXECUTION VALIDITY",
        "verdict": "PASS" if citable else "FAIL",
        "citable": citable,
        "values": {
            key: {"value": value, "unit": unit, "trace_id": f"t-{key}"}
            for key, (value, unit) in values.items()
        },
    }


def config(tmp_path, **kwargs) -> Gate2Config:
    return Gate2Config(artifact_root=str(tmp_path), **kwargs)


# --------------------------------------------------------------------------- #
# entry conditions
# --------------------------------------------------------------------------- #


def test_rejected_registry_is_not_checkable(tmp_path):
    """Gate 1's rejection stands; Gate 2 must not be reachable past it."""
    reg = registry({"exp1.acc": (0.81, "ratio")}, citable=False)
    with pytest.raises(GateError, match="Gate 1 did not pass"):
        run_gate2(reg, config(tmp_path))


def test_report_is_written_to_the_attempt_dir(tmp_path):
    report = run_gate2(registry({"exp1.acc": (0.81, "ratio")}), config(tmp_path))
    path = tmp_path / "gate2" / "attempt_01" / "gate2_report.json"
    written = json.loads(path.read_text())
    assert written["gate"] == GATE_NAME
    assert written["verdict"] == "PASS"
    assert report.artifact_dir == str(tmp_path / "gate2" / "attempt_01")


# --------------------------------------------------------------------------- #
# coherence.range_valid
# --------------------------------------------------------------------------- #


def test_accuracy_above_one_fails(tmp_path):
    reg = registry({"exp1.acc": (1.04, "ratio")})
    report = run_gate2(reg, config(tmp_path))
    assert report.verdict is Verdict.FAIL
    check = next(c for c in report.checks if c.id == "coherence.range_valid")
    assert check.evidence["violations"][0]["key"] == "exp1.acc"


def test_negative_loss_fails(tmp_path):
    report = run_gate2(registry({"exp1.loss": (-0.2, "loss")}), config(tmp_path))
    assert report.verdict is Verdict.FAIL


def test_zero_loss_passes_but_zero_wallclock_does_not(tmp_path):
    """`loss >= 0` and `time > 0` differ, and the difference is load-bearing.

    A wallclock of exactly zero is an unmeasured run, not a fast one.
    """
    assert run_gate2(registry({"a.loss": (0.0, "loss")}), config(tmp_path)).passed
    reg = registry({"a.wallclock_s": (0.0, "seconds")})
    assert not run_gate2(reg, config(tmp_path)).passed


def test_value_with_unknown_unit_is_reported_unchecked_not_passed(tmp_path):
    """Coverage is reported, so "in range" is never confused with "unchecked"."""
    reg = registry({"a.acc": (0.8, "ratio"), "a.weird": (999.0, "furlongs")})
    report = run_gate2(reg, config(tmp_path))
    check = next(c for c in report.checks if c.id == "coherence.range_valid")
    assert report.passed
    assert check.evidence["checked"] == 1
    assert check.evidence["unchecked"] == ["a.weird"]


def test_unit_matching_ignores_case_and_padding(tmp_path):
    reg = registry({"a.acc": (1.5, " Ratio ")})
    assert not run_gate2(reg, config(tmp_path)).passed


def test_per_key_override_beats_the_unit_table(tmp_path):
    reg = registry({"a.acc": (0.8, "ratio")})
    cfg = config(tmp_path, ranges={"a.acc": Range(0.9, 1.0)})
    assert not run_gate2(reg, cfg).passed


def test_non_numeric_values_are_skipped(tmp_path):
    reg = registry({"a.note": ("done", "ratio"), "a.flag": (True, "ratio")})
    report = run_gate2(reg, config(tmp_path))
    check = next(c for c in report.checks if c.id == "coherence.range_valid")
    assert report.passed
    assert check.evidence["checked"] == 0


def test_a_nan_value_cannot_pass_as_a_measurement(tmp_path):
    """NaN is not a value in range; it is the absence of one.

    ``Range.admits`` rejects NaN only where an upper bound exists, because
    ``value <= high`` is what NaN fails. Every unbounded-above unit — the
    timings, the counts, the losses, the speedups — admitted it.
    """
    for unit in ("seconds", "loss", "count", "speedup", "ms", "wallclock_s"):
        reg = registry({"a.x": (float("nan"), unit)})
        report = run_gate2(reg, config(tmp_path))
        assert not report.passed, f"NaN passed range_valid for unit {unit!r}"


def test_an_infinite_value_cannot_pass_as_a_measurement(tmp_path):
    for unit in ("seconds", "loss", "count", "speedup"):
        reg = registry({"a.x": (float("inf"), unit)})
        assert not run_gate2(reg, config(tmp_path)).passed, unit


def test_a_speedup_divided_by_an_unmeasured_wallclock_cannot_pass(tmp_path):
    """The case the two halves of Gate 2 were already half-guarding.

    ``low_open`` exists so a wallclock of exactly zero is rejected as unmeasured.
    ``OPS["ratio"]`` returns NaN when its denominator is zero. So the gate caught
    the unmeasured time and then admitted the speedup derived from it.
    """
    reg = registry({
        "a.baseline_s": (13.61, "seconds"),
        "a.ours_s": (0.0, "seconds"),
        "a.speedup": (float("nan"), "speedup"),
    })
    report = run_gate2(reg, config(tmp_path))
    check = next(c for c in report.checks if c.id == "coherence.range_valid")
    flagged = {v["key"] for v in check.evidence["violations"]}
    assert "a.speedup" in flagged, "the NaN speedup was not flagged"


def test_a_non_finite_violation_names_the_value_as_non_finite(tmp_path):
    """The feedback has to say what is wrong, not just that something is.

    "nan lies outside (0, +inf]" is not an instruction anybody can act on.
    """
    reg = registry({"a.t": (float("nan"), "seconds")})
    report = run_gate2(reg, config(tmp_path))
    check = next(c for c in report.checks if c.id == "coherence.range_valid")
    assert any("not a finite number" in d for d in check.evidence["discrepancies"])


def test_perplexity_below_one_cannot_survive(tmp_path):
    """Perplexity is exp(H) and cross entropy cannot be negative.

    This bound is arithmetic, not convention, so it is the one range in the
    table that can be claimed as elimination by construction.
    """
    assert not run_gate2(registry({"a.ppl": (0.3, "perplexity")}), config(tmp_path)).passed
    assert run_gate2(registry({"a.ppl": (1.0, "perplexity")}), config(tmp_path)).passed
    assert run_gate2(registry({"a.ppl": (42.7, "perplexity")}), config(tmp_path)).passed


def test_an_auc_above_one_cannot_survive(tmp_path):
    assert not run_gate2(registry({"a.auc": (1.02, "auc")}), config(tmp_path)).passed
    assert not run_gate2(registry({"a.auc": (-0.1, "auc")}), config(tmp_path)).passed
    assert run_gate2(registry({"a.auc": (0.87, "auc")}), config(tmp_path)).passed


@pytest.mark.parametrize("unit", ["f1", "precision", "recall"])
def test_a_classification_score_above_one_cannot_survive(tmp_path, unit):
    assert not run_gate2(registry({"a.m": (1.5, unit)}), config(tmp_path)).passed
    assert run_gate2(registry({"a.m": (0.5, unit)}), config(tmp_path)).passed


def test_a_metric_name_alone_never_implies_a_range(tmp_path):
    """The bounds are keyed on the declared unit, never on the metric's name.

    Guessing from the name would reject a legitimately negative score that some
    scaffold happens to call ``f1``. A value whose unit this gate does not know
    stays *unchecked*, which the report says out loud.
    """
    reg = registry({"a.f1": (1.5, None), "a.precision": (-3.0, "furlongs")})
    report = run_gate2(reg, config(tmp_path))
    check = next(c for c in report.checks if c.id == "coherence.range_valid")
    assert report.passed
    assert check.evidence["checked"] == 0
    assert check.evidence["unchecked"] == ["a.f1", "a.precision"]


# --------------------------------------------------------------------------- #
# coherence.plausibility
# --------------------------------------------------------------------------- #


def speedup_registry(value, **extra):
    reg = registry({"a.baseline_s": (4700.0, "seconds"), "a.ours_s": (1.0, "seconds"),
                    "a.speedup": (value, "speedup")})
    reg.update(extra)
    return reg


DERIVES_SPEEDUP = Relation(key="a.speedup", op="ratio", left="a.baseline_s", right="a.ours_s")


def test_an_unexplained_speedup_above_the_ceiling_cannot_survive(tmp_path):
    """A large speedup nothing derives is a reporting defect, not a result."""
    report = run_gate2(speedup_registry(4700.0), config(tmp_path))
    assert not report.passed
    check = next(c for c in report.checks if c.id == "coherence.plausibility")
    assert check.severity is Severity.FAIL
    assert check.evidence["violations"][0]["key"] == "a.speedup"


def test_a_speedup_a_declared_relation_derives_is_exempt_at_any_magnitude(tmp_path):
    """4,700x is a real published ratio, so the ceiling must not reject it.

    SAGE (arXiv 2606.31478) reports an FVA runtime about 4,700x FBA. The gate
    asks for the arithmetic, not for a smaller number: once a relation derives
    the value from two recorded times, the magnitude stops being evidence.
    """
    cfg = config(tmp_path, relations=(DERIVES_SPEEDUP,))
    report = run_gate2(speedup_registry(4700.0), cfg)
    assert report.passed
    check = next(c for c in report.checks if c.id == "coherence.plausibility")
    assert check.evidence["exempt"] == ["a.speedup"]


def test_a_speedup_whose_declared_relation_does_not_hold_is_not_exempt(tmp_path):
    """A relation that fails cannot buy an exemption it did not earn."""
    cfg = config(tmp_path, relations=(DERIVES_SPEEDUP,))
    report = run_gate2(speedup_registry(9999.0), cfg)
    check = next(c for c in report.checks if c.id == "coherence.plausibility")
    assert not check.passed
    assert check.evidence["exempt"] == []


def test_plausibility_emits_no_check_when_nothing_recorded_a_speedup(tmp_path):
    """Absent, never green: no speedup means no opinion about speedups."""
    report = run_gate2(registry({"a.acc": (0.81, "ratio")}), config(tmp_path))
    assert [c for c in report.checks if c.id == "coherence.plausibility"] == []


def test_the_declared_ceiling_is_recorded_in_the_report(tmp_path):
    """A hand-chosen bound appears in the evidence, not only in the source.

    A reviewer has to be able to disagree with the number rather than guess at
    it, which is the same reason ``Band.origin`` exists.
    """
    report = run_gate2(speedup_registry(4700.0), config(tmp_path, implausible_speedup=2500.0))
    check = next(c for c in report.checks if c.id == "coherence.plausibility")
    assert check.evidence["ceiling"] == 2500.0
    assert check.evidence["ceiling_origin"] == "declared"

    # The shipped default travels the same way, so the report never leaves a
    # reviewer to infer which bound was applied.
    default = run_gate2(speedup_registry(4700.0), config(tmp_path))
    assert next(
        c for c in default.checks if c.id == "coherence.plausibility"
    ).evidence["ceiling"] == IMPLAUSIBLE_SPEEDUP


def test_plausibility_feedback_asks_for_the_relation_not_a_smaller_number(tmp_path):
    """The remedy is to show the arithmetic, and the text has to say so."""
    report = run_gate2(speedup_registry(4700.0), config(tmp_path))
    check = next(c for c in report.checks if c.id == "coherence.plausibility")
    assert any("relation" in d for d in check.evidence["discrepancies"])


def test_a_non_finite_speedup_is_left_to_the_range_check(tmp_path):
    """One defect, one check, one fix.

    ``range_valid`` owns nan and inf. A registry whose only speedup is
    non-finite leaves the plausibility check nothing it can compare, so it emits
    no check at all rather than a pass that would read as "the speedups here are
    fine".
    """
    report = run_gate2(speedup_registry(float("inf")), config(tmp_path))
    assert not report.passed
    assert [c for c in report.checks if c.id == "coherence.plausibility"] == []
    ranges = next(c for c in report.checks if c.id == "coherence.range_valid")
    assert ranges.evidence["violations"][0]["key"] == "a.speedup"


def test_a_finite_speedup_is_still_checked_beside_a_non_finite_one(tmp_path):
    """Dropping the unusable subject must not drop the usable one with it."""
    reg = speedup_registry(float("nan"))
    reg["values"]["a.other_speedup"] = {
        "value": 5000.0, "unit": "speedup", "trace_id": "t-other"
    }
    report = run_gate2(reg, config(tmp_path))
    check = next(c for c in report.checks if c.id == "coherence.plausibility")
    assert check.evidence["checked"] == 1
    assert check.evidence["violations"][0]["key"] == "a.other_speedup"


def test_the_plausibility_finding_reaches_the_agent(tmp_path):
    """Half two of the feedback loop: the finding has to render, not just exist.

    ``render_evidence`` dispatches on a per-id registry, so a check added
    without an entry produces a report that names the failure and withholds the
    fix. A rejection the agent cannot act on is a stall, not a gate.
    """
    report = run_gate2(speedup_registry(4700.0), config(tmp_path))
    check = next(c for c in report.checks if c.id == "coherence.plausibility")
    # The message alone already names the number, so assert on the two things
    # only a registered renderer and a registered directive can produce.
    assert render_evidence(check), "the check has no evidence renderer"
    text = render_feedback(report)
    assert "no declared relation derives it" in text
    assert "REQUIRED FIXES" in text
    assert "the arithmetic is on the record" in text


def test_every_check_gate2_emits_can_be_rendered_and_has_a_fix(tmp_path):
    """Structural guard over the two registries in ``gates.report``.

    Keyed lookups fail silently when a key is missing, so a new check ships a
    report with no evidence rows and no required fix, and nothing complains.
    That is how ``coherence.plausibility`` first shipped.
    """
    from gates.report import _EVIDENCE_RENDERERS, _FIXES

    reg = registry({
        "a.acc": (1.4, "ratio"),
        "a.baseline_s": (4700.0, "seconds"),
        "a.ours_s": (1.0, "seconds"),
        "a.speedup": (9999.0, "speedup"),
    })
    cfg = config(tmp_path, relations=(DERIVES_SPEEDUP,), sources=(WU2019,))
    emitted = {check.id for check in run_gate2(reg, cfg).checks}

    assert emitted, "the fixture emitted no checks at all"
    assert sorted(i for i in emitted if i not in _EVIDENCE_RENDERERS) == []
    assert sorted(i for i in emitted if i not in _FIXES) == []


# --------------------------------------------------------------------------- #
# coherence.internal_consistency
# --------------------------------------------------------------------------- #


SPEEDUP = Relation(
    key="exp2.speedup", op="ratio", left="exp2.gcn.wallclock_s", right="exp2.sgc.wallclock_s"
)


def test_declared_speedup_that_matches_the_recorded_times_holds(tmp_path):
    """`PLAN.md` §4.3's worked example: 0.2450 / 0.0180 = 13.61x."""
    reg = registry(
        {
            "exp2.speedup": (13.611, "speedup"),
            "exp2.gcn.wallclock_s": (0.2450, "seconds"),
            "exp2.sgc.wallclock_s": (0.0180, "seconds"),
        }
    )
    report = run_gate2(reg, config(tmp_path, relations=(SPEEDUP,)))
    assert report.passed
    check = next(c for c in report.checks if c.id == "coherence.internal_consistency")
    assert check.evidence["held"] == 1


def test_speedup_that_contradicts_the_recorded_times_fails(tmp_path):
    reg = registry(
        {
            "exp2.speedup": (42.0, "speedup"),
            "exp2.gcn.wallclock_s": (0.2450, "seconds"),
            "exp2.sgc.wallclock_s": (0.0180, "seconds"),
        }
    )
    report = run_gate2(reg, config(tmp_path, relations=(SPEEDUP,)))
    assert report.verdict is Verdict.FAIL
    check = next(c for c in report.checks if c.id == "coherence.internal_consistency")
    failure = check.evidence["failures"][0]
    assert failure["recorded"] == 42.0
    assert failure["expected"] == pytest.approx(13.611, rel=1e-3)


def test_a_relation_whose_operands_are_missing_is_unresolved_not_passed(tmp_path):
    """Otherwise a plan collects a green check for a relation it never recorded."""
    reg = registry({"exp2.speedup": (13.611, "speedup")})
    report = run_gate2(reg, config(tmp_path, relations=(SPEEDUP,)))
    assert report.verdict is Verdict.FAIL
    check = next(c for c in report.checks if c.id == "coherence.internal_consistency")
    assert check.evidence["unresolved"][0]["missing"] == [
        "exp2.gcn.wallclock_s",
        "exp2.sgc.wallclock_s",
    ]


def test_division_by_a_recorded_zero_fails_rather_than_raising(tmp_path):
    reg = registry(
        {
            "exp2.speedup": (13.611, "speedup"),
            "exp2.gcn.wallclock_s": (0.2450, "seconds"),
            "exp2.sgc.wallclock_s": (0.0, "seconds"),
        }
    )
    report = run_gate2(reg, config(tmp_path, relations=(SPEEDUP,)))
    assert report.verdict is Verdict.FAIL


def test_rounding_the_agent_applied_is_absorbed_by_the_tolerance(tmp_path):
    """13.61 reported to two decimals must not be called a contradiction."""
    reg = registry(
        {
            "exp2.speedup": (13.61, "speedup"),
            "exp2.gcn.wallclock_s": (0.2450, "seconds"),
            "exp2.sgc.wallclock_s": (0.0180, "seconds"),
        }
    )
    relation = Relation(SPEEDUP.key, SPEEDUP.op, SPEEDUP.left, SPEEDUP.right, rel_tol=1e-3)
    assert run_gate2(reg, config(tmp_path, relations=(relation,))).passed


def test_unknown_op_is_a_wiring_error(tmp_path):
    reg = registry({"a": (1.0, "ratio"), "b": (1.0, "ratio"), "c": (1.0, "ratio")})
    bad = Relation(key="a", op="quotient", left="b", right="c")
    with pytest.raises(GateError, match="unknown relation op"):
        run_gate2(reg, config(tmp_path, relations=(bad,)))


def test_the_other_three_ops(tmp_path):
    # No unit the range table knows: this test is about the ops, and a sum of
    # two ratios leaving [0, 1] would fail the other check for unrelated reasons.
    reg = registry(
        {
            "d": (0.5, None),
            "s": (1.5, None),
            "p": (0.5, None),
            "a": (1.0, None),
            "b": (0.5, None),
        }
    )
    relations = (
        Relation("d", "difference", "a", "b"),
        Relation("s", "sum", "a", "b"),
        Relation("p", "product", "a", "b"),
    )
    report = run_gate2(reg, config(tmp_path, relations=relations))
    assert report.passed


# --------------------------------------------------------------------------- #
# exhaustion carries forward rather than blocking
# --------------------------------------------------------------------------- #


def test_discrepancies_survive_for_gate_3_to_require_declared(tmp_path):
    reg = registry(
        {
            "exp1.acc": (1.04, "ratio"),
            "exp2.speedup": (42.0, "speedup"),
            "exp2.gcn.wallclock_s": (0.2450, "seconds"),
            "exp2.sgc.wallclock_s": (0.0180, "seconds"),
        }
    )
    report = run_gate2(reg, config(tmp_path, relations=(SPEEDUP,)))
    carried = unresolved_discrepancies(report)
    assert any("exp1.acc" in line for line in carried)
    assert any("exp2.speedup" in line for line in carried)


def test_a_passing_report_carries_nothing_forward(tmp_path):
    report = run_gate2(registry({"exp1.acc": (0.81, "ratio")}), config(tmp_path))
    assert unresolved_discrepancies(report) == []


def test_gate2_default_budget_is_two(tmp_path):
    """`PLAN.md` §4: two revisions, then proceed with the discrepancy declared."""
    assert run_gate2(registry({}), config(tmp_path)).max_attempts == 2


# --------------------------------------------------------------------------- #
# tier B — coherence.reference_interval
# --------------------------------------------------------------------------- #


def reference(report):
    return next(
        (c for c in report.checks if c.id == "coherence.reference_interval"), None
    )


WU2019 = SourceClaim(
    key="exp1.K2.test_acc",
    source_id="arXiv:1902.07153",
    value=0.810,
    interval=(0.800, 0.820),
    setting="SGC on Cora, K=2",
)


def test_b_does_not_run_without_a_corpus(tmp_path):
    """Tier A alone emits no literature check — absent, never green.

    A reported check that never ran is the failure mode a `tiers` flag would
    have introduced.
    """
    report = run_gate2(registry({"exp1.K2.test_acc": (0.81, "ratio")}), config(tmp_path))
    assert reference(report) is None
    assert [c.id for c in report.checks] == [
        "coherence.range_valid",
        "coherence.internal_consistency",
    ]


def test_value_inside_the_reported_interval_agrees(tmp_path):
    reg = registry({"exp1.K2.test_acc": (0.812, "ratio")})
    report = run_gate2(reg, config(tmp_path, sources=(WU2019,)))
    check = reference(report)
    assert check.passed
    assert check.evidence["agreed"][0]["band_origin"] == "reported_interval"
    assert check.evidence["band_origins"] == {"reported_interval": 1}


def test_value_outside_the_reported_interval_is_a_discrepancy(tmp_path):
    reg = registry({"exp1.K2.test_acc": (0.923, "ratio")})
    report = run_gate2(reg, config(tmp_path, sources=(WU2019,)))
    check = reference(report)
    assert not check.passed
    assert check.evidence["out_of_band"][0]["value"] == 0.923


def test_b_warns_by_default_and_fails_only_in_strict_mode(tmp_path):
    """`PLAN.md` §4.2: WARN → FAIL in strict mode. The verdict must follow."""
    reg = registry({"exp1.K2.test_acc": (0.923, "ratio")})
    lenient = run_gate2(reg, config(tmp_path, sources=(WU2019,)))
    assert lenient.verdict is Verdict.PASS
    assert reference(lenient).severity is Severity.WARN

    strict = run_gate2(
        reg, config(tmp_path, sources=(WU2019,), strict_reference=True), attempt=2
    )
    assert strict.verdict is Verdict.FAIL
    assert reference(strict).severity is Severity.FAIL


def test_q1_an_unreferenced_result_is_new_not_wrong(tmp_path):
    """Q1. No source covers K=8, so the gate must not call the number wrong.

    It says so, carries it forward, and passes — in strict mode too.
    """
    reg = registry(
        {"exp1.K2.test_acc": (0.812, "ratio"), "exp3.noloop.K8.test_acc": (0.392, "ratio")}
    )
    report = run_gate2(
        reg, config(tmp_path, sources=(WU2019,), strict_reference=True)
    )
    check = reference(report)
    assert check.passed and report.verdict is Verdict.PASS
    assert check.evidence["unreferenced"] == ["exp3.noloop.K8.test_acc"]
    assert any(
        "novel rather than as a replication" in d
        for d in unresolved_discrepancies(report)
    )


def test_q2_a_point_estimate_uses_a_declared_tolerance_and_says_so(tmp_path):
    """Q2. A band nobody published is labelled, never passed off as principled."""
    point = SourceClaim(
        key="exp1.K2.test_acc", source_id="arXiv:1902.07153", value=0.810, rel_tol=0.02
    )
    reg = registry({"exp1.K2.test_acc": (0.820, "ratio")})
    report = run_gate2(reg, config(tmp_path, sources=(point,)))
    check = reference(report)
    assert check.passed
    assert check.evidence["agreed"][0]["band_origin"] == "declared_relative"


def test_q2_a_source_with_no_tolerance_at_all_falls_back_and_is_labelled(tmp_path):
    bare = SourceClaim(key="a.acc", source_id="arXiv:1902.07153", value=0.800)
    reg = registry({"a.acc": (0.820, "ratio")})
    report = run_gate2(reg, config(tmp_path, sources=(bare,), default_rel_tol=0.05))
    assert reference(report).evidence["band_origins"] == {"default_relative": 1}


def test_band_precedence_is_interval_then_declared_then_default():
    both = SourceClaim(key="k", source_id="s", value=1.0, interval=(0.9, 1.1), rel_tol=0.5)
    assert band_for(both, 0.05).origin == "reported_interval"
    declared = SourceClaim(key="k", source_id="s", value=1.0, rel_tol=0.5)
    assert band_for(declared, 0.05).origin == "declared_relative"
    assert band_for(SourceClaim(key="k", source_id="s", value=1.0), 0.05) == Band(
        0.95, 1.05, "default_relative"
    )


def test_agreeing_with_one_of_two_disagreeing_sources_is_enough(tmp_path):
    """Two papers disagreeing is the literature's problem, not the run's."""
    a = SourceClaim(key="a.acc", source_id="paper-a", value=0.81, interval=(0.80, 0.82))
    b = SourceClaim(key="a.acc", source_id="paper-b", value=0.75, interval=(0.74, 0.76))
    reg = registry({"a.acc": (0.815, "ratio")})
    report = run_gate2(reg, config(tmp_path, sources=(a, b), strict_reference=True))
    check = reference(report)
    assert check.passed
    assert check.evidence["agreed"][0]["source_id"] == "paper-a"


def test_a_source_number_the_run_never_measured_is_reported_not_failed(tmp_path):
    reg = registry({"exp1.K2.test_acc": (0.812, "ratio")})
    extra = SourceClaim(key="exp9.missing", source_id="paper-c", value=0.5)
    report = run_gate2(
        reg, config(tmp_path, sources=(WU2019, extra), strict_reference=True)
    )
    check = reference(report)
    assert check.passed
    assert check.evidence["unmeasured"] == [
        {"key": "exp9.missing", "source_id": "paper-c"}
    ]


def test_an_inverted_interval_is_normalised(tmp_path):
    flipped = SourceClaim(key="a.acc", source_id="s", value=0.81, interval=(0.82, 0.80))
    assert band_for(flipped, 0.05) == Band(0.80, 0.82, "reported_interval")


# --------------------------------------------------------------------------- #
# tier C — the semantic tier
# --------------------------------------------------------------------------- #


def model_returning(*payloads):
    """A fake model that answers each call with the next payload in turn."""
    replies = list(payloads)
    calls = []

    def fn(prompt, system=""):
        calls.append((prompt, system))
        return replies.pop(0) if replies else "[]"

    fn.calls = calls
    return fn


def semantic(report, check_id):
    return next((c for c in report.checks if c.id == check_id), None)


def test_c_does_not_run_without_a_model(tmp_path):
    report = run_gate2(registry({"a.acc": (0.8, "ratio")}), config(tmp_path))
    assert semantic(report, "coherence.method_match") is None
    assert semantic(report, "coherence.claim_supported") is None
    assert report.model is None


def test_method_mismatch_is_reported_as_a_warning(tmp_path):
    fn = model_returning(
        '[{"source_id": "arXiv:1902.07153", "aspect": "row-normalised adjacency",'
        ' "why": "source uses symmetric normalisation, which changes the accuracy"}]',
        "[]",
    )
    reg = registry({"exp1.K2.test_acc": (0.812, "ratio")})
    report = run_gate2(
        reg,
        config(
            tmp_path,
            sources=(WU2019,),
            consult_model=fn,
            method_source="A = row_normalise(adj)",
            claims=("SGC matches GCN accuracy on Cora",),
        ),
    )
    check = semantic(report, "coherence.method_match")
    assert check.severity is Severity.WARN
    assert check.evidence["findings"][0]["ref"] == "arXiv:1902.07153"


def test_q3_the_semantic_tier_cannot_move_the_verdict(tmp_path):
    """Q3. BadScientist puts this at ≈ chance, so it must decide nothing.

    Both passes flag everything they are shown; the verdict stays PASS.
    """
    fn = model_returning(
        '[{"source_id": "arXiv:1902.07153", "aspect": "different splits", "why": "x"}]',
        '[{"claim": 0, "why": "a trend cannot be established from one point"}]',
    )
    reg = registry({"exp1.K2.test_acc": (0.812, "ratio")})
    report = run_gate2(
        reg,
        config(
            tmp_path,
            sources=(WU2019,),
            consult_model=fn,
            method_source="whatever",
            claims=("accuracy degrades with depth",),
        ),
    )
    assert report.verdict is Verdict.PASS
    assert all(
        c.severity is not Severity.FAIL
        for c in report.checks
        if c.id.startswith("coherence.method") or c.id.startswith("coherence.claim")
    )


def test_severity_fail_is_absent_from_the_semantic_module():
    """The structural guarantee, asserted the way Gate 1 asserts its own."""
    source = (pathlib.Path(gates.gate2_semantic.__file__)).read_text()
    body = source.split('"""', 2)[-1]
    assert "Severity.FAIL" not in body


def test_an_ungrounded_source_id_is_discarded(tmp_path):
    """A finding citing a paper nobody retrieved must never reach the engineer."""
    fn = model_returning(
        '[{"source_id": "arXiv:0000.00000", "aspect": "invented", "why": "x"}]', "[]"
    )
    reg = registry({"exp1.K2.test_acc": (0.812, "ratio")})
    report = run_gate2(
        reg,
        config(tmp_path, sources=(WU2019,), consult_model=fn, method_source="code"),
    )
    assert semantic(report, "coherence.method_match") is None


def test_an_out_of_range_claim_index_is_discarded(tmp_path):
    fn = model_returning("[]", '[{"claim": 7, "why": "unsupported"}]')
    reg = registry({"a.acc": (0.8, "ratio")})
    report = run_gate2(
        tmp_path and reg,
        config(tmp_path, consult_model=fn, claims=("only one claim",)),
    )
    assert semantic(report, "coherence.claim_supported") is None


def test_a_model_that_fails_degrades_to_info_and_records_it(tmp_path):
    def boom(prompt, system=""):
        raise RuntimeError("ollama is down")

    reg = registry({"a.acc": (0.8, "ratio")})
    report = run_gate2(
        reg,
        config(
            tmp_path,
            sources=(WU2019,),
            consult_model=boom,
            method_source="code",
            claims=("a claim",),
        ),
    )
    check = semantic(report, "coherence.method_match")
    assert check.severity is Severity.INFO and check.passed
    assert check.evidence["degraded"] is True
    assert report.verdict is Verdict.PASS
    assert report.model["failures"] == 2


def test_prose_wrapped_json_is_still_parsed(tmp_path):
    """Small models fence the array however firmly the prompt says not to."""
    fn = model_returning(
        'Sure!\n```json\n[{"source_id": "arXiv:1902.07153", "aspect": "splits",'
        ' "why": "different"}]\n```',
        "[]",
    )
    reg = registry({"exp1.K2.test_acc": (0.812, "ratio")})
    report = run_gate2(
        reg, config(tmp_path, sources=(WU2019,), consult_model=fn, method_source="c")
    )
    assert semantic(report, "coherence.method_match") is not None


def test_the_model_is_never_shown_a_value_the_registry_lacks(tmp_path):
    fn = model_returning("[]", "[]")
    reg = registry({"a.acc": (0.8, "ratio")})
    run_gate2(
        reg,
        config(
            tmp_path,
            sources=(WU2019,),
            consult_model=fn,
            method_source="code",
            claims=("a claim",),
        ),
    )
    claim_prompt = fn.calls[1][0]
    assert "a.acc = 0.8" in claim_prompt
    assert "exp1.K2.test_acc" not in claim_prompt


# --------------------------------------------------------------------------- #
# the tier combinations the slides ask for
# --------------------------------------------------------------------------- #


CLEAN = {
    "exp1.K2.test_acc": (0.812, "ratio"),
    "exp2.speedup": (13.611, "speedup"),
    "exp2.gcn.wallclock_s": (0.2450, "seconds"),
    "exp2.sgc.wallclock_s": (0.0180, "seconds"),
}


def tier_config(tmp_path, *, b: bool, c: bool):
    quiet = model_returning("[]", "[]")
    return config(
        tmp_path,
        relations=(SPEEDUP,),
        sources=(WU2019,) if b else (),
        consult_model=quiet if c else None,
        method_source="A = normalise(adj)" if c else "",
        claims=("SGC matches GCN on Cora",) if c else (),
    )


@pytest.mark.parametrize(
    "b, c, expected",
    [
        (
            False,
            False,
            [
                "coherence.range_valid",
                "coherence.internal_consistency",
                "coherence.plausibility",
            ],
        ),
        (
            True,
            False,
            [
                "coherence.range_valid",
                "coherence.internal_consistency",
                "coherence.plausibility",
                "coherence.reference_interval",
            ],
        ),
        (
            False,
            True,
            [
                "coherence.range_valid",
                "coherence.internal_consistency",
                "coherence.plausibility",
                # A+C cannot compare a method to sources it was not given. The
                # pass records that it did not run instead of staying silent.
                "coherence.method_match",
            ],
        ),
        (
            True,
            True,
            [
                "coherence.range_valid",
                "coherence.internal_consistency",
                "coherence.plausibility",
                "coherence.reference_interval",
            ],
        ),
    ],
    ids=["A", "A+B", "A+C", "A+B+C"],
)
def test_each_tier_combination_emits_exactly_its_own_checks(tmp_path, b, c, expected):
    """A / A+B / A+C / A+B+C, on a clean registry.

    Two behaviours this pins down:

    * Tier C contributes no check when it *ran* and found nothing — silence, not
      a green row. So A+B+C lists the same ids as A+B on a clean registry.
    * ``plausibility`` is in every list because the clean registry records a
      speedup, so the check has a subject. It is tier A with an input
      condition, not a tier of its own: a registry with no speedup omits it.
    * **A+C is not a whole configuration.** ``method_match`` asks whether the
      implementation matches what the cited source describes, which is not a
      question without cited sources, so it degrades to INFO and says so. Tier C
      depends on tier B's corpus for half of its work.
    """
    report = run_gate2(registry(CLEAN), tier_config(tmp_path, b=b, c=c))
    assert [check.id for check in report.checks] == expected
    assert report.passed


@pytest.mark.parametrize("b, c", [(False, False), (True, False), (False, True), (True, True)])
def test_no_combination_lets_a_tier_a_violation_through(tmp_path, b, c):
    """Whatever else is switched on, A still decides. The gate is the authority."""
    broken = dict(CLEAN, **{"exp1.K2.test_acc": (1.4, "ratio")})
    report = run_gate2(registry(broken), tier_config(tmp_path, b=b, c=c))
    assert report.verdict is Verdict.FAIL


@pytest.mark.parametrize(
    "b, c, calls",
    [(False, False, None), (True, False, None), (False, True, 1), (True, True, 2)],
    ids=["A", "A+B", "A+C", "A+B+C"],
)
def test_the_model_is_called_only_in_the_c_combinations(tmp_path, b, c, calls):
    """Tier C costs two calls per attempt, one without a corpus, zero when off.

    The A+C row is one call rather than two because ``method_match`` declines
    before spending anything: a pass with no source to compare against does not
    get billed for asking.
    """
    report = run_gate2(registry(CLEAN), tier_config(tmp_path, b=b, c=c))
    if calls is None:
        assert report.model is None
    else:
        assert report.model["calls"] == calls
        assert report.model["degraded"] is False


def test_a_plus_c_reports_method_match_as_unavailable_not_clean(tmp_path):
    """The A+C degradation, stated as its own assertion rather than a list diff."""
    report = run_gate2(registry(CLEAN), tier_config(tmp_path, b=False, c=True))
    check = semantic(report, "coherence.method_match")
    assert check.severity is Severity.INFO and check.evidence["degraded"] is True
    assert "no cited source" in check.evidence["error"]


# --------------------------------------------------------------------------- #
# the feedback report
# --------------------------------------------------------------------------- #


OUT_OF_BAND = {"exp1.K2.test_acc": (0.951, "ratio")}
UNREFERENCED = {"exp1.K8.test_acc": (0.788, "ratio")}


def test_feedback_does_not_claim_the_run_never_happened(tmp_path):
    """Gate 2 judges a run that already finished.

    ``render_feedback`` printed "rejected before execution" for any report
    without an ``execution``, which is every Gate 2 report. The guard is
    ``code_sha256``, which Gate 1 sets on both of its paths and Gate 2 never
    sets.
    """
    reg = registry({"exp1.acc": (1.4, "ratio")})
    text = render_feedback(run_gate2(reg, config(tmp_path)))
    assert "nothing was run" not in text


def test_range_violation_renders_the_offending_value(tmp_path):
    reg = registry({"exp1.acc": (1.4, "ratio")})
    text = render_feedback(run_gate2(reg, config(tmp_path)))
    assert "exp1.acc = 1.4" in text
    assert "[0, 1]" in text  # Range.describe uses %g, so 0.0 renders as 0


def test_relation_failure_renders_both_operands(tmp_path):
    """The agent cannot recompute a ratio it cannot see the inputs to."""
    reg = registry(
        {
            "exp2.speedup": (42.0, "speedup"),
            "exp2.gcn.wallclock_s": (0.2450, "seconds"),
            "exp2.sgc.wallclock_s": (0.0180, "seconds"),
        }
    )
    text = render_feedback(run_gate2(reg, config(tmp_path, relations=(SPEEDUP,))))
    assert "exp2.gcn.wallclock_s=0.245" in text
    assert "exp2.sgc.wallclock_s=0.018" in text


def test_unresolved_relation_says_which_operand_was_missing(tmp_path):
    reg = registry({"exp2.speedup": (13.6, "speedup")})
    text = render_feedback(run_gate2(reg, config(tmp_path, relations=(SPEEDUP,))))
    assert "nothing recorded for" in text
    assert "exp2.gcn.wallclock_s" in text


def test_out_of_band_renders_the_source_and_where_its_band_came_from(tmp_path):
    """Q2 in the feedback report: the band's provenance travels with the band."""
    report = run_gate2(
        registry(OUT_OF_BAND),
        config(tmp_path, sources=(WU2019,), strict_reference=True),
    )
    text = render_feedback(report)
    assert "arXiv:1902.07153" in text
    assert "reported_interval" in text
    assert "SGC on Cora, K=2" in text


def test_an_unreferenced_result_reaches_the_writer(tmp_path):
    """Q1 in the loop, not just in the report JSON.

    The reference check *passes* when it only found unreferenced keys, so it is
    in neither ``failed_checks`` nor ``warnings`` and would render nothing. That
    would drop the one instruction the wrong/new split exists to produce.
    """
    report = run_gate2(registry(UNREFERENCED), config(tmp_path, sources=(WU2019,)))
    assert report.verdict is Verdict.PASS
    text = render_feedback(report)
    assert "CARRY FORWARD" in text
    assert "exp1.K8.test_acc" in text
    assert "novel rather than as a replication" in text


def test_semantic_findings_render_in_the_deterministic_shape(tmp_path):
    """A finding from a model must not look different from a measured one."""
    fn = model_returning(
        '[{"source_id": "arXiv:1902.07153", "aspect": "row-normalized adjacency",'
        ' "why": "changes the propagation matrix and the reported accuracy"}]'
    )
    report = run_gate2(
        registry({"exp1.K2.test_acc": (0.812, "ratio")}),
        config(
            tmp_path,
            sources=(WU2019,),
            consult_model=fn,
            method_source="A = adj / adj.sum(1)",
        ),
    )
    text = render_feedback(report)
    assert "arXiv:1902.07153: row-normalized adjacency" in text
    assert "changes the propagation matrix" in text


def test_required_fixes_are_offered_for_the_blocking_checks(tmp_path):
    reg = registry({"exp1.acc": (1.4, "ratio")})
    text = render_feedback(run_gate2(reg, config(tmp_path)))
    assert "REQUIRED FIXES" in text
    assert "the unit it actually has" in text


def test_no_fix_directive_exists_for_a_check_that_cannot_block():
    """A fix the agent can never be shown is dead code, not caution.

    Both semantic checks are built through ``model_warning``, so neither can
    ever be ``blocking``, so neither can ever reach ``_required_fixes``.
    """
    from gates.report import _FIXES

    assert "coherence.method_match" not in _FIXES
    assert "coherence.claim_supported" not in _FIXES


# --------------------------------------------------------------------------- #
# the adapter: gate 2 driven the way gate 1 is
# --------------------------------------------------------------------------- #


def test_the_host_wiring_path_actually_reaches_gate_2(tmp_path):
    """B7: ``make_context`` builds a Gate1Config, which ``run_gate2`` cannot use.

    Gate 2 looked wired because ``gated_review`` exists, and every test built its
    own ``Gate2Config`` by hand. Driven the way a host is documented to drive it,
    the call raised ``AttributeError: 'Gate1Config' object has no attribute
    'ranges'``. A gate with no reachable entry point is not installed.
    """
    ctx = make_review_context(research_dir=str(tmp_path))
    assert isinstance(ctx.config, Gate2Config)

    reg = registry({"exp1.acc": (0.81, "ratio")})
    result = gated_review(reg, ctx)
    assert result.report.verdict is Verdict.PASS


def test_the_review_context_carries_gate_2_configuration(tmp_path):
    """Whatever the host declares has to survive into the verdict.

    A context that accepted relations and ranges and then dropped them would
    fail open, which is worse than not accepting them.
    """
    ctx = make_review_context(
        research_dir=str(tmp_path),
        relations=(DERIVES_SPEEDUP,),
        ranges={"a.acc": Range(0.9, 1.0)},
        implausible_speedup=250.0,
    )
    assert ctx.config.relations == (DERIVES_SPEEDUP,)
    assert ctx.config.implausible_speedup == 250.0

    # The range override is the cheapest one to prove end to end: 0.81 is a
    # legal ratio and an illegal one under the declared override.
    result = gated_review(registry({"a.acc": (0.81, "ratio")}), ctx)
    assert result.report.verdict is Verdict.FAIL


def test_the_review_budget_is_two_attempts_not_gate_ones_three(tmp_path):
    """Gate 2 gets two revisions per PLAN.md 4, Gate 1 gets three.

    They are different budgets for different jobs, and a shared default would
    silently give Gate 2 an extra turn it was never allotted.
    """
    assert make_review_context(research_dir=str(tmp_path)).config.max_attempts == 2


def test_gated_review_counts_agent_turns_not_executions(tmp_path):
    """The feedback header must move when the agent gets another turn.

    Without ``rewrite``, ``render_feedback`` reads "attempt 1 of 2" forever,
    because ``attempt`` is the artifact ordinal rather than the budget.
    """
    ctx = GateContext(config=config(tmp_path))
    reg = registry({"exp1.acc": (1.4, "ratio")})

    first = gated_review(reg, ctx)
    assert first.report.rewrite == 1
    ctx.close_turn(passed=False)

    second = gated_review(reg, ctx)
    assert second.report.rewrite == 2
    assert second.report.attempt == 2


def test_gated_review_carries_limitations_into_the_writing_phase(tmp_path):
    """Gate 2 proceeds, so what it found has to travel rather than stop the run."""
    ctx = GateContext(config=config(tmp_path, sources=(WU2019,)))
    outcome = gated_review(registry(UNREFERENCED), ctx)

    assert outcome.passed
    assert "DECLARED LIMITATIONS" in outcome.evidence_bundle
    assert "novel rather than as a replication" in outcome.evidence_bundle


def test_a_clean_review_declares_nothing(tmp_path):
    ctx = GateContext(config=config(tmp_path))
    outcome = gated_review(registry({"exp1.acc": (0.81, "ratio")}), ctx)
    assert declared_limitations(outcome.report) == ""


def test_an_exhausted_budget_is_labelled_by_the_gate_that_spent_it(tmp_path):
    """The gate name was hardcoded to Gate 1, so any gate raised Gate 1's failure.

    Gate 2 never calls this - its policy is to proceed - but the context is
    shared, and a mislabelled failure sends whoever reads the log to the wrong
    gate.
    """
    ctx = GateContext(config=config(tmp_path))
    reg = registry({"exp1.acc": (1.4, "ratio")})
    for _ in range(2):
        gated_review(reg, ctx)
        ctx.close_turn(passed=False)

    assert ctx.budget_exhausted
    with pytest.raises(GateFailure) as excinfo:
        ctx.check_can_continue()
    assert excinfo.value.gate == GATE_NAME
