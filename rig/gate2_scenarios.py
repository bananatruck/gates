"""Gate 2 feedback-loop scenarios: the registries a scripted engineer submits.

Each turn is the registry a re-run wrote after the engineer read the previous
feedback. In a host, a Gate 2 rejection sends the engineer back through Gate 1 to
produce that registry; a scenario supplies it directly, because Gate 2's contract
is the registry format and a scenario that executed an experiment to reach Gate 2
would test Gate 1 twice.

The five scenarios are `GATE2_implementation_spec.md` §3 C4, with scenario 4
restated for D17. The spec wrote it as a divergence the engineer justifies and
Gate 2 only warns on. Divergence is FAIL since D17, so a justification changes
nothing and that story is scenario 5. Gate 2's one WARN-and-proceed path is a
declared field nobody can check, which is what scenario 4 now plays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gates.gate2 import PlanField, Relation

#: The relation `PLAN.md` §4.3 uses as its worked example.
SPEEDUP = Relation(
    key="exp2.speedup", op="ratio", left="exp2.gcn.wallclock_s", right="exp2.sgc.wallclock_s"
)
LR = PlanField(key="config.lr", declared=0.001, source_span="plan L4: learning rate 0.001")
EPOCHS = PlanField(key="config.epochs", declared=200, source_span="plan L5: 200 epochs")
DROPOUT = PlanField(key="config.dropout", declared=0.5, source_span="plan L6: dropout 0.5")

#: key -> (value, unit, provenance.arg_kind). A run that did what its plan said.
CLEAN: dict[str, tuple[Any, str | None, str]] = {
    "exp1.acc": (0.812, "ratio", "computed"),
    "exp2.gcn.wallclock_s": (0.245, "seconds", "computed"),
    "exp2.sgc.wallclock_s": (0.018, "seconds", "computed"),
    "exp2.speedup": (13.611, "speedup", "computed"),
    "config.lr": (0.001, None, "constant"),
    "config.epochs": (200, "count", "constant"),
}


@dataclass(frozen=True)
class Turn:
    """One engineer turn: the registry it submitted, and what Gate 2 must do."""

    label: str
    values: dict[str, tuple[Any, str | None, str]]
    #: Check ids that must appear among the blocking failures.
    expect_fail: tuple[str, ...] = ()
    #: Check ids that must appear among the warnings.
    expect_warn: tuple[str, ...] = ()
    expect_pass: bool = False

    def registry(self) -> dict[str, Any]:
        """The registry, in the shape ``gates.registry.build_registry`` emits."""
        return {
            "gate": "GATE 1 — EXECUTION VALIDITY",
            "verdict": "PASS",
            "citable": True,
            "values": {
                key: {"value": value, "unit": unit, "provenance": {"arg_kind": kind}}
                for key, (value, unit, kind) in self.values.items()
            },
        }


@dataclass(frozen=True)
class Scenario:
    name: str
    summary: str
    turns: tuple[Turn, ...]
    #: "pass" - a registry was admitted. "proceeded" - the budget was spent with
    #: a discrepancy unresolved, so the run continues and declares it.
    expect_outcome: str
    expect_turns: int
    max_attempts: int = 2
    relations: tuple[Relation, ...] = (SPEEDUP,)
    plan_fields: tuple[PlanField, ...] = (LR, EPOCHS)
    #: Text that must reach the writer in the declared limitations.
    expect_declared: tuple[str, ...] = ()


CLEAN_RUN = Scenario(
    name="clean",
    summary="The run did what the plan said. Admitted first time, nothing declared.",
    turns=(Turn("conforming run", CLEAN, expect_pass=True),),
    expect_outcome="pass",
    expect_turns=1,
)

OUT_OF_RANGE = Scenario(
    name="out-of-range",
    summary="An accuracy of 1.4 is rejected; the re-run records a real one.",
    turns=(
        Turn(
            "accuracy recorded as 1.4",
            {**CLEAN, "exp1.acc": (1.4, "ratio", "computed")},
            expect_fail=("coherence.range_valid",),
        ),
        Turn("re-run records 0.812", CLEAN, expect_pass=True),
    ),
    expect_outcome="pass",
    expect_turns=2,
)

BROKEN_RELATION = Scenario(
    name="broken-relation",
    summary="A speedup its own two wallclocks contradict is rejected, then corrected.",
    turns=(
        Turn(
            "speedup recorded as 42x against 0.245 s / 0.018 s",
            {**CLEAN, "exp2.speedup": (42.0, "speedup", "computed")},
            expect_fail=("coherence.internal_consistency",),
        ),
        Turn("speedup recomputed from the wallclocks", CLEAN, expect_pass=True),
    ),
    expect_outcome="pass",
    expect_turns=2,
)

UNVERIFIABLE_PLAN = Scenario(
    name="unverifiable-plan",
    summary="Two declared fields nobody can check. WARN, proceeds, both declared.",
    turns=(
        Turn(
            "lr typed at the record_result call, dropout never recorded",
            {**CLEAN, "config.lr": (0.001, None, "literal")},
            expect_warn=("coherence.method_traceable",),
            expect_pass=True,
        ),
    ),
    expect_outcome="pass",
    expect_turns=1,
    plan_fields=(LR, EPOCHS, DROPOUT),
    expect_declared=("typed at the record_result call", "never recorded it"),
)

DIVERGENCE_EXHAUSTS = Scenario(
    name="divergence-exhausts",
    summary="The run used another learning rate and never fixes it. Proceeds, declared.",
    turns=(
        Turn(
            "run used lr 0.01",
            {**CLEAN, "config.lr": (0.01, None, "constant")},
            expect_fail=("coherence.method_conformance",),
        ),
        Turn(
            "engineer argues for 0.01 instead of re-running",
            {**CLEAN, "config.lr": (0.01, None, "constant")},
            expect_fail=("coherence.method_conformance",),
        ),
    ),
    expect_outcome="proceeded",
    expect_turns=2,
    expect_declared=("the plan declared 0.001 and the run recorded 0.01",),
)

SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in (DIVERGENCE_EXHAUSTS, CLEAN_RUN, OUT_OF_RANGE, BROKEN_RELATION, UNVERIFIABLE_PLAN)
}
