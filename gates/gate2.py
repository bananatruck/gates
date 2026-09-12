"""Gate 2 — source ↔ result coherence.

Answers: *are these measured results consistent with what the cited literature
reports for this method, dataset and setting?*

Gate 2 runs on Gate 1's registry, not on source. By the time it fires the
numbers already exist and are already attributable to one execution; the
question is no longer whether they were produced but whether they can be
believed. That is why its input is `registry.json` and not a program.

Two tiers, and the difference between them is the honesty boundary
(`PLAN.md` §4.4):

* **Deterministic.** Provable, and claimable as elimination-by-construction. A
  value outside its unit's admissible range, or a declared arithmetic relation
  that does not hold, is a fact about the numbers alone.
* **Semantic.** Model-assisted, reported as a rate with an interval, never as a
  guarantee. Not built yet.

Three tiers, and which of them run is decided by **what the caller supplies**,
not by a flag:

    A  coherence.range_valid           FAIL   always
       coherence.internal_consistency  FAIL   always
    B  coherence.reference_interval    WARN   iff `sources` were supplied
                                       FAIL   ...and `strict_reference` is set
    C  coherence.method_match          WARN   iff `consult_model` was supplied
       coherence.claim_supported       WARN   ""

Deriving the tier from its input rather than from a `tiers` flag removes the
one failure mode a flag introduces: a run configured for tier B with no corpus
loaded, reporting a green literature check it never performed. A tier with no
input does not emit a check at all — absent, never green.

`PLAN.md` §4.4's honesty boundary falls exactly on the A/B–C line: A and B are
provable and may be claimed as elimination-by-construction; C may only be
reported as a rate with an interval.

Gate 2's exhaustion behaviour differs from Gates 1 and 3 and the difference is
deliberate. Gate 1 refuses to hand on a run that did not happen; Gate 3 refuses
to emit an unverifiable manuscript. Gate 2 *proceeds* — an unresolved
discrepancy with the literature is a limitation to declare, not grounds to
discard a real measurement. Carrying it forward is `unresolved_discrepancies`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import gate2_semantic
from .errors import GateError
from .llm import (
    DEFAULT_MAX_PROMPT_CHARS,
    DEFAULT_TIMEOUT_S as DEFAULT_MODEL_TIMEOUT_S,
    ModelFn,
    ModelLayer,
)
from .schema import CheckResult, GateReport, Severity, Verdict, decide

GATE_NAME = "GATE 2 — SOURCE ↔ RESULT COHERENCE"


@dataclass(frozen=True)
class Range:
    """An admissible interval for a unit. ``None`` bound means unbounded."""

    low: float | None = None
    high: float | None = None
    #: True for "time > 0" as against "loss >= 0". The distinction is the whole
    #: point of the check for durations: a wallclock of exactly 0.0 is not a
    #: fast run, it is an unmeasured one.
    low_open: bool = False

    def admits(self, value: float) -> bool:
        # NaN and the infinities are rejected before either bound is consulted.
        # Without this the check depends on an accident: `value <= high` is the
        # comparison NaN fails, so a bounded unit rejected it and every
        # unbounded-above unit — the timings, counts, losses and speedups —
        # admitted it. A division by an unmeasured wallclock produces exactly
        # that NaN, which is the case `low_open` already refuses one step
        # earlier. Neither is a measurement, so neither is in range.
        if not math.isfinite(value):
            return False
        if self.low is not None:
            if value < self.low or (self.low_open and value == self.low):
                return False
        return self.high is None or value <= self.high

    def describe(self) -> str:
        lo = "-inf" if self.low is None else f"{self.low:g}"
        hi = "+inf" if self.high is None else f"{self.high:g}"
        return f"{'(' if self.low_open else '['}{lo}, {hi}]"


#: Unit → admissible range. Keyed on the ``unit`` string the experiment passed to
#: ``record_result``, lowercased. Deliberately small and deliberately literal:
#: guessing a range from a metric's *name* would fail a legitimately negative
#: score named ``..._acc`` by a scaffold that means something else by it, and a
#: false rejection costs the engineer a rewrite for nothing.
UNIT_RANGES: dict[str, Range] = {
    "ratio": Range(0.0, 1.0),
    "fraction": Range(0.0, 1.0),
    "proportion": Range(0.0, 1.0),
    "probability": Range(0.0, 1.0),
    "accuracy": Range(0.0, 1.0),
    "percent": Range(0.0, 100.0),
    "percentage": Range(0.0, 100.0),
    "loss": Range(low=0.0),
    "count": Range(low=0.0),
    "s": Range(low=0.0, low_open=True),
    "sec": Range(low=0.0, low_open=True),
    "secs": Range(low=0.0, low_open=True),
    "seconds": Range(low=0.0, low_open=True),
    "ms": Range(low=0.0, low_open=True),
    "wallclock_s": Range(low=0.0, low_open=True),
    "speedup": Range(low=0.0, low_open=True),
    # Scores whose definition bounds them, added as units rather than as metric
    # names. An author who writes unit="f1" has declared what the number is; a
    # table keyed on the name would instead be guessing, which is what the note
    # above rules out. A metric this table does not know stays unchecked, and
    # the report says so.
    "auc": Range(0.0, 1.0),
    "f1": Range(0.0, 1.0),
    "precision": Range(0.0, 1.0),
    "recall": Range(0.0, 1.0),
    # Perplexity is exp(H) and cross entropy cannot be negative, so ppl >= 1 is
    # arithmetic rather than convention. It is the only entry here that can be
    # claimed as elimination by construction without further argument.
    "perplexity": Range(low=1.0),
}

#: The speedup above which a value no declared relation derives is treated as a
#: measurement or reporting defect rather than a result.
#:
#: Declared, not derived, and the distinction is the whole point. No argument
#: makes 1000 the right number; it is a judgement someone made, so it travels
#: into the report as ``ceiling_origin: "declared"`` and a reviewer can move it.
#: The bound is deliberately not in ``UNIT_RANGES``: everything in that table is
#: a fact about the numbers, and mixing a prior into it would weaken the
#: elimination-by-construction claim that ``coherence.range_valid`` supports.
#:
#: SAGE (arXiv 2606.31478) reports a real FVA runtime about 4,700x FBA, which is
#: why magnitude alone cannot be the test. A value two recorded measurements
#: derive is exempt at any size.
IMPLAUSIBLE_SPEEDUP = 1000.0

#: The arithmetic a plan can declare between recorded values. Enough for the
#: relation `PLAN.md` §4.3 gives as the worked example — a speedup that must
#: equal the ratio of two recorded times — and nothing more. An expression
#: language was rejected: it needs a safe evaluator, and no plan has yet
#: declared a relation these four cannot state.
OPS = {
    "ratio": lambda a, b: a / b if b else math.nan,
    "difference": lambda a, b: a - b,
    "sum": lambda a, b: a + b,
    "product": lambda a, b: a * b,
}


@dataclass(frozen=True)
class Relation:
    """A declared arithmetic identity between registry keys.

    ``key`` must equal ``op(left, right)`` to within ``rel_tol``. The relation is
    part of the *plan*, not of the results: the engineer says what a derived
    number means, and Gate 2 checks that it means it.
    """

    key: str
    op: str
    left: str
    right: str
    #: Relative tolerance. Defaults loose enough to absorb the rounding an agent
    #: applies when it reports a derived figure to two decimals.
    rel_tol: float = 1e-3

    def compute(self, left: float, right: float) -> float:
        try:
            return OPS[self.op](left, right)
        except KeyError:
            raise GateError(
                f"unknown relation op {self.op!r}; known: {', '.join(sorted(OPS))}"
            ) from None


@dataclass(frozen=True)
class SourceClaim:
    """One number a cited source reports, for one metric key and setting.

    This is Gate 2's view of the retrieved literature, and it is deliberately
    not the scaffold's. The host's corpus (`phd.lit_review`: arXiv ids plus full
    text) is a different and much larger object; an adapter extracts these from
    it, exactly as `gates/adapters/` carries every other piece of host knowledge.

    ``interval`` is the principled band and ``rel_tol`` the declared fallback —
    see :func:`band_for` for why the difference is recorded rather than smoothed
    over.
    """

    #: The registry key this source number bears on.
    key: str
    #: arXiv id or bib key. Gate 3 requires this to be in the retrieval
    #: registry, so a band can never come from a paper nobody fetched.
    source_id: str
    value: float
    #: The prediction interval the source itself reports, if it reports one.
    interval: tuple[float, float] | None = None
    #: Relative tolerance declared for this claim, when the source gives only a
    #: point estimate.
    rel_tol: float | None = None
    #: Human description of the setting compared, for the feedback report.
    setting: str = ""
    #: What the source says it does, for tier C's method comparison.
    describes: str = ""


@dataclass(frozen=True)
class Band:
    """A tolerance band, carrying where it came from.

    ``origin`` is load-bearing rather than diagnostic. A band taken from a
    reported prediction interval and a band invented by a default are not the
    same evidence, and a paper that reports "within tolerance" without saying
    which it used has overclaimed. This is `PLAN.md` §4.2's requirement that a
    hand-chosen band "is recorded as such", made a field.
    """

    low: float
    high: float
    #: "reported_interval" | "declared_relative" | "default_relative"
    origin: str

    def admits(self, value: float) -> bool:
        return self.low <= value <= self.high

    def describe(self) -> str:
        return f"[{self.low:g}, {self.high:g}]"


def band_for(claim: SourceClaim, default_rel_tol: float) -> Band:
    """The band for one source claim, most principled option first.

    CORE-Bench's methodology is the reason for the ordering: a tolerance derived
    from a reported prediction interval is a property of the source, while a
    relative tolerance is a decision someone made. Both are usable; only the
    first is defensible without further argument, so the second is labelled.
    """
    if claim.interval is not None:
        low, high = sorted(claim.interval)
        return Band(low, high, "reported_interval")
    if claim.rel_tol is not None:
        margin = abs(claim.value) * claim.rel_tol
        return Band(claim.value - margin, claim.value + margin, "declared_relative")
    margin = abs(claim.value) * default_rel_tol
    return Band(claim.value - margin, claim.value + margin, "default_relative")


@dataclass
class Gate2Config:
    """Everything Gate 2 needs. No scaffold types, same as Gate 1."""

    #: Revisions the ML engineer gets before Gate 2 gives up. Two, per
    #: `PLAN.md` §4 — and giving up means proceeding with the discrepancy
    #: declared, not refusing to continue.
    max_attempts: int = 2
    #: Arithmetic identities the plan declared between recorded values.
    relations: tuple[Relation, ...] = ()
    #: Per-key range overrides, for a metric whose unit the table does not know
    #: or whose admissible range is narrower than its unit's.
    ranges: dict[str, Range] = field(default_factory=dict)
    #: The speedup above which an *underived* value is treated as a defect. A
    #: config field rather than a bare constant so the number reaches the
    #: report, where a reviewer can disagree with it instead of guessing at it.
    implausible_speedup: float = IMPLAUSIBLE_SPEEDUP
    artifact_root: str = "gate_artifacts"

    # -- tier B ------------------------------------------------------------- #
    #: What the cited literature reports. Empty means tier B does not run, and
    #: emits no check.
    sources: tuple[SourceClaim, ...] = ()
    #: Band for a claim that supplies neither an interval nor its own tolerance.
    #: Recorded on every such band as ``default_relative``.
    default_rel_tol: float = 0.05
    #: `PLAN.md` §4.2: reference_interval is WARN, "→ FAIL in strict mode".
    strict_reference: bool = False

    # -- tier C ------------------------------------------------------------- #
    #: The model. Absent means tier C does not run, and emits no check.
    consult_model: ModelFn | None = None
    #: The implementation tier C compares against the sources — normally the
    #: experiment source Gate 1 already executed.
    method_source: str = ""
    #: The claims the plan says these results establish.
    claims: tuple[str, ...] = ()
    model_timeout_s: float = DEFAULT_MODEL_TIMEOUT_S
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS

    def attempt_dir(self, attempt: int) -> Path:
        return Path(self.artifact_root) / "gate2" / f"attempt_{attempt:02d}"


def run_gate2(
    registry: dict[str, Any], config: Gate2Config, attempt: int = 1
) -> GateReport:
    """Run Gate 2's deterministic tier against a Gate 1 registry.

    Raises ``GateError`` if the registry is not citable. That is a wiring
    defect, not an agent mistake: no rewrite the engineer makes can turn a
    rejected run into a citable one, so failing the gate would send it into a
    loop it cannot exit. Invariant 2 — the gate is the authority — means Gate 2
    must not be reachable with an artifact Gate 1 rejected.
    """
    if not registry.get("citable"):
        raise GateError(
            "Gate 2 was handed a registry Gate 1 did not pass "
            f"(verdict {registry.get('verdict')!r}). Gate 1's rejection stands; "
            "there is nothing here to check for coherence."
        )

    values = registry.get("values") or {}

    # Tier A — always. Deterministic, and the only tier that can fail a run on
    # its own evidence.
    checks = [
        _check_range_valid(values, config),
        _check_internal_consistency(values, config),
    ]

    # Tier A, but only with a subject. A declared ceiling is still deterministic;
    # it just has nothing to say about a registry that recorded no speedup.
    plausibility = _check_plausibility(values, config)
    if plausibility is not None:
        checks.append(plausibility)

    # Tier B — only with a corpus. Deterministic once a band exists; the
    # judgement is in where the band came from, which the check records.
    reference = _check_reference_interval(values, config)
    if reference is not None:
        checks.append(reference)

    # Tier C — only with a model, and WARN-only by construction. Appended before
    # decide() purely because grouping the report by tier reads better than
    # grouping it by author; decide() is blind to everything here either way.
    model = ModelLayer(
        config.consult_model,
        timeout_s=config.model_timeout_s,
        max_prompt_chars=config.max_prompt_chars,
    )
    if model.available:
        checks.extend(_semantic_checks(model, values, config))

    artifact_dir = config.attempt_dir(attempt)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    report = GateReport(
        gate=GATE_NAME,
        # Fixed from tier A and tier B alone. Tier C is WARN by construction and
        # cannot reach this line — the same guarantee Gate 1 makes, for the same
        # reason, and here it also carries BadScientist's near-chance detection
        # rate, which is why C must never decide anything.
        verdict=decide(checks),
        attempt=attempt,
        max_attempts=config.max_attempts,
        checks=checks,
        artifact_dir=str(artifact_dir),
        model=model.budget.to_dict() if model.available else None,
    )
    (artifact_dir / "gate2_report.json").write_text(
        report.to_json(), encoding="utf-8"
    )
    return report


def unresolved_discrepancies(report: GateReport) -> list[str]:
    """What an exhausted Gate 2 carries forward for Gate 3 to require stated.

    Gate 2 proceeding is not Gate 2 forgetting. Every discrepancy it could not
    get resolved becomes a limitation the manuscript is obliged to declare.

    Passing checks are read too, and that is the point rather than an oversight.
    Q1's answer — an unreferenced result is NEW, not WRONG — makes a result with
    no comparable source pass the gate while still needing to be declared. If
    only failing checks were harvested, the one case the distinction exists for
    would be the one case that never reached the manuscript.
    """
    out: list[str] = []
    for check in report.checks:
        declared = check.evidence.get("discrepancies") or []
        if declared:
            out.extend(declared)
        elif not check.passed:
            out.append(check.message)
    return out


# --------------------------------------------------------------------------- #
# the deterministic tier
# --------------------------------------------------------------------------- #


def _describe_violation(v: dict[str, Any]) -> str:
    """One line of feedback for one out-of-range value.

    Split by cause. A value that overshot a bound needs the bound quoted so the
    engineer can see by how much; a non-finite value needs no bound at all,
    because the defect is upstream of the range.
    """
    if not v["finite"]:
        return (
            f"{v['key']} = {v['value']!r} is not a finite number, so it is not "
            f"a measurement; check the computation that produced it "
            f"(a division by an unmeasured value yields nan)"
        )
    return (
        f"{v['key']} = {v['value']!r} lies outside {v['range']} "
        f"for unit {v['unit']!r}"
    )


def _check_range_valid(
    values: dict[str, Any], config: Gate2Config
) -> CheckResult:
    """Every value with a known unit lies inside that unit's range.

    Coverage is reported alongside the verdict. A value whose unit the table
    does not know is *unchecked*, and saying how many were unchecked is the
    difference between "these numbers are in range" and "these numbers are not
    known to be out of range".
    """
    violations: list[dict[str, Any]] = []
    checked = 0
    unchecked: list[str] = []

    for key, entry in values.items():
        value = entry.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        allowed = _range_for(key, entry, config)
        if allowed is None:
            unchecked.append(key)
            continue
        checked += 1
        if not allowed.admits(float(value)):
            violations.append(
                {
                    "key": key,
                    "value": value,
                    "unit": entry.get("unit"),
                    "range": allowed.describe(),
                    # A non-finite value failed for a different reason than a
                    # value that overshot a bound, and the feedback has to say
                    # which. "nan lies outside (0, +inf]" tells the engineer
                    # nothing they can act on.
                    "finite": math.isfinite(float(value)),
                }
            )

    if violations:
        first = violations[0]
        message = (
            f"{len(violations)} value(s) outside their unit's admissible range, "
            f"e.g. {first['key']} = {first['value']!r} "
            f"(unit {first['unit']!r}, admissible {first['range']})"
        )
    else:
        message = (
            f"{checked} value(s) inside their unit's admissible range; "
            f"{len(unchecked)} carried no unit this gate knows"
        )

    return CheckResult(
        id="coherence.range_valid",
        passed=not violations,
        severity=Severity.FAIL,
        message=message,
        evidence={
            "violations": violations,
            "checked": checked,
            "unchecked": sorted(unchecked),
            "discrepancies": [_describe_violation(v) for v in violations],
        },
    )


def _relation_holds(values: dict[str, Any], relation: Relation) -> bool:
    """Whether every operand resolves and the declared identity holds.

    A yes-or-no reading of the same arithmetic ``_check_internal_consistency``
    reports on in detail. Kept separate because that check has to distinguish a
    relation that failed from one whose operands were never recorded, and here
    both answers are the same: nothing was derived.
    """
    operands: dict[str, float] = {}
    for role, key in (("key", relation.key), ("left", relation.left), ("right", relation.right)):
        value = _numeric(values.get(key))
        if value is None:
            return False
        operands[role] = value
    expected = relation.compute(operands["left"], operands["right"])
    if math.isnan(expected):
        return False
    return math.isclose(operands["key"], expected, rel_tol=relation.rel_tol)


def _check_plausibility(
    values: dict[str, Any], config: Gate2Config
) -> CheckResult | None:
    """A speedup above the declared ceiling that no declared relation derives.

    Separate from ``range_valid`` on purpose. A unit's admissible range is a
    fact about the numbers; a ceiling is a prior about what results occur. Both
    are useful and only one is provable, so they get separate ids and the report
    says which kind of finding it is carrying.

    The gate is provenance, not magnitude. A speedup two recorded times derive
    passes at any size, because the arithmetic is on the record. The remedy for
    a violation is therefore to declare the relation, not to report a smaller
    number, and the feedback says so.

    Returns ``None`` when nothing recorded a speedup. A gate with no speedups in
    front of it has no opinion about speedups, and an opinion it never formed
    must not render as a passing check.
    """
    subjects: dict[str, float] = {}
    for key, entry in values.items():
        unit = entry.get("unit")
        if not isinstance(unit, str) or unit.strip().lower() != "speedup":
            continue
        value = entry.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        # nan and inf belong to range_valid. One defect, one check, one fix.
        if math.isfinite(float(value)):
            subjects[key] = float(value)

    if not subjects:
        return None

    ceiling = config.implausible_speedup
    derived = {r.key for r in config.relations if _relation_holds(values, r)}
    over = {k: v for k, v in subjects.items() if v > ceiling}
    exempt = sorted(k for k in over if k in derived)
    violations = [
        {"key": k, "value": v, "ceiling": ceiling}
        for k, v in sorted(over.items())
        if k not in derived
    ]

    if violations:
        first = violations[0]
        message = (
            f"{len(violations)} speedup(s) above the declared ceiling of "
            f"{ceiling:g}x that no declared relation derives, "
            f"e.g. {first['key']} = {first['value']:g}x"
        )
    else:
        message = (
            f"{len(subjects)} speedup(s) checked against a declared ceiling of "
            f"{ceiling:g}x; {len(exempt)} above it and derived by a declared relation"
        )

    return CheckResult(
        id="coherence.plausibility",
        passed=not violations,
        severity=Severity.FAIL,
        message=message,
        evidence={
            "violations": violations,
            "exempt": exempt,
            "checked": len(subjects),
            "ceiling": ceiling,
            # Says out loud that this bound was chosen rather than computed, for
            # the same reason Band.origin does.
            "ceiling_origin": "declared",
            "discrepancies": [
                f"{v['key']} = {v['value']:g}x is above the declared ceiling of "
                f"{v['ceiling']:g}x and no declared relation derives it; declare the "
                f"relation that computes it from the recorded measurements, or "
                f"correct the value"
                for v in violations
            ],
        },
    )


def _check_internal_consistency(
    values: dict[str, Any], config: Gate2Config
) -> CheckResult:
    """Every declared arithmetic relation holds, to its tolerance.

    A relation whose operands are missing is reported unresolved rather than
    passed. Silently skipping it would let a plan declare a relation, record
    neither operand, and collect a green check for it.
    """
    failures: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    held = 0

    for relation in config.relations:
        operands = {}
        missing = []
        for role, key in (("key", relation.key), ("left", relation.left), ("right", relation.right)):
            value = _numeric(values.get(key))
            if value is None:
                missing.append(key)
            else:
                operands[role] = value
        if missing:
            unresolved.append(
                {"key": relation.key, "op": relation.op, "missing": missing}
            )
            continue

        expected = relation.compute(operands["left"], operands["right"])
        actual = operands["key"]
        if math.isnan(expected) or not math.isclose(
            actual, expected, rel_tol=relation.rel_tol
        ):
            failures.append(
                {
                    "key": relation.key,
                    "recorded": actual,
                    "expected": expected,
                    "op": relation.op,
                    "left": relation.left,
                    "left_value": operands["left"],
                    "right": relation.right,
                    "right_value": operands["right"],
                    "rel_tol": relation.rel_tol,
                }
            )
        else:
            held += 1

    problems = failures + unresolved
    if failures:
        first = failures[0]
        message = (
            f"{len(failures)} declared relation(s) do not hold, e.g. "
            f"{first['key']} recorded as {first['recorded']!r} but "
            f"{first['op']}({first['left']}, {first['right']}) = "
            f"{first['expected']!r}"
        )
    elif unresolved:
        message = (
            f"{len(unresolved)} declared relation(s) could not be checked: "
            f"operands missing from the registry"
        )
    else:
        message = f"{held} declared relation(s) hold"

    return CheckResult(
        id="coherence.internal_consistency",
        passed=not problems,
        severity=Severity.FAIL,
        message=message,
        evidence={
            "failures": failures,
            "unresolved": unresolved,
            "held": held,
            "discrepancies": [
                f"{f['key']} recorded as {f['recorded']!r}, but "
                f"{f['left']}={f['left_value']!r} and "
                f"{f['right']}={f['right_value']!r} give {f['expected']!r}"
                for f in failures
            ]
            + [
                f"relation on {u['key']} could not be checked: "
                f"{', '.join(u['missing'])} not in the registry"
                for u in unresolved
            ],
        },
    )


# --------------------------------------------------------------------------- #
# tier B — the reference interval
# --------------------------------------------------------------------------- #


def _check_reference_interval(
    values: dict[str, Any], config: Gate2Config
) -> CheckResult | None:
    """Measured values against what a comparable source reports.

    Two outcomes, kept apart, because conflating them is Q1 — *how do we
    distinguish WRONG from NEW?*

    * **Out of band.** A source reports a comparable number and the measurement
      is outside its tolerance. That is a discrepancy, and it can fail the check.
    * **Unreferenced.** No source in the retrieved corpus reports anything for
      this key. That is *not* evidence of error; it is a result with no
      comparison available. It never fails the check, is always reported, and is
      carried forward so the manuscript must call the result novel rather than
      pass it off as a replication — `PLAN.md` §4.3's worked example verbatim.

    A key with several sources needs to agree with only one of them. Sources
    disagreeing with each other is the literature's business, not the
    experiment's, and demanding agreement with all of them would fail a correct
    measurement whenever two papers disagree.
    """
    if not config.sources:
        return None

    by_key: dict[str, list[SourceClaim]] = {}
    for claim in config.sources:
        by_key.setdefault(claim.key, []).append(claim)

    out_of_band: list[dict[str, Any]] = []
    agreed: list[dict[str, Any]] = []
    unreferenced: list[str] = []
    unmeasured: list[dict[str, str]] = []

    for key, entry in values.items():
        value = _numeric(entry)
        if value is None:
            continue
        claims = by_key.get(key)
        if not claims:
            unreferenced.append(key)
            continue
        bands = [(c, band_for(c, config.default_rel_tol)) for c in claims]
        match = next((c_b for c_b in bands if c_b[1].admits(value)), None)
        if match is not None:
            claim, band = match
            agreed.append(
                {
                    "key": key,
                    "value": value,
                    "source_id": claim.source_id,
                    "band": band.describe(),
                    "band_origin": band.origin,
                }
            )
        else:
            out_of_band.append(
                {
                    "key": key,
                    "value": value,
                    "candidates": [
                        {
                            "source_id": c.source_id,
                            "reported": c.value,
                            "band": b.describe(),
                            "band_origin": b.origin,
                            "setting": c.setting,
                        }
                        for c, b in bands
                    ],
                }
            )

    # A source number for a key the run never recorded. Not a failure — the
    # corpus is allowed to be wider than the experiment — but worth saying, since
    # it is usually a plan that dropped a comparison it promised.
    for key, claims in by_key.items():
        if key not in values:
            unmeasured.extend(
                {"key": key, "source_id": c.source_id} for c in claims
            )

    severity = Severity.FAIL if config.strict_reference else Severity.WARN
    if out_of_band:
        first = out_of_band[0]
        candidate = first["candidates"][0]
        message = (
            f"{len(out_of_band)} value(s) outside every comparable source's "
            f"tolerance band, e.g. {first['key']} = {first['value']!r} against "
            f"{candidate['source_id']} {candidate['band']} "
            f"({candidate['band_origin']})"
        )
    else:
        message = (
            f"{len(agreed)} value(s) agree with a cited source; "
            f"{len(unreferenced)} have no comparable source in the corpus"
        )

    return CheckResult(
        id="coherence.reference_interval",
        # Only a real disagreement can fail this. An unreferenced result is a
        # result nobody has published a comparison for, which is what a novel
        # finding looks like from inside the gate.
        passed=not out_of_band,
        severity=severity,
        message=message,
        evidence={
            "out_of_band": out_of_band,
            "agreed": agreed,
            "unreferenced": sorted(unreferenced),
            "unmeasured": unmeasured,
            "band_origins": _band_origin_counts(agreed, out_of_band),
            "strict": config.strict_reference,
            "discrepancies": [
                f"{v['key']} = {v['value']!r} falls outside "
                + ", ".join(
                    f"{c['source_id']} {c['band']}" for c in v["candidates"]
                )
                for v in out_of_band
            ]
            + [
                f"{key} has no comparable number in the retrieved corpus — cite a "
                f"source that reports this setting, or state the result as novel "
                f"rather than as a replication"
                for key in sorted(unreferenced)
            ],
        },
    )


def _band_origin_counts(
    agreed: list[dict[str, Any]], out_of_band: list[dict[str, Any]]
) -> dict[str, int]:
    """How many bands came from a reported interval versus a declared tolerance.

    The paper needs this number. "Within tolerance" means something different
    when the tolerance was published than when it was chosen, and a rate is the
    only honest way to say which this run had.
    """
    counts: dict[str, int] = {}
    for row in agreed:
        counts[row["band_origin"]] = counts.get(row["band_origin"], 0) + 1
    for row in out_of_band:
        for candidate in row["candidates"]:
            origin = candidate["band_origin"]
            counts[origin] = counts.get(origin, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# tier C — the semantic tier
# --------------------------------------------------------------------------- #


def _semantic_checks(
    model: ModelLayer, values: dict[str, Any], config: Gate2Config
) -> list[CheckResult]:
    """Tier C's checks, or nothing when a pass had no input and found nothing.

    Every result here is WARN or INFO by construction; see ``gate2_semantic``.
    """
    sources = [
        {
            "source_id": c.source_id,
            "describes": c.describes,
            "setting": c.setting,
        }
        for c in config.sources
    ]
    produced = [
        gate2_semantic.build_method_check(
            gate2_semantic.scan_method_match(
                model, config.method_source, sources
            )
        ),
        gate2_semantic.build_claim_check(
            gate2_semantic.scan_claims_supported(model, list(config.claims), values)
        ),
    ]
    return [c for c in produced if c is not None]


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _range_for(
    key: str, entry: dict[str, Any], config: Gate2Config
) -> Range | None:
    if key in config.ranges:
        return config.ranges[key]
    unit = entry.get("unit")
    if not isinstance(unit, str):
        return None
    return UNIT_RANGES.get(unit.strip().lower())


def _numeric(entry: Any) -> float | None:
    if not isinstance(entry, dict):
        return None
    value = entry.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None
