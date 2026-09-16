"""Gate 3 feedback-loop scenarios: the manuscripts a scripted writer submits.

Each turn is the manuscript the writer submitted after reading the previous
feedback, with its ``\\result{}`` tokens intact. Every token names a key from
Gate 2's ``CLEAN`` table, because the registry these manuscripts are judged
against comes from playing Gate 2's ``clean`` scenario, not from a hand-built
dict (F12).

Numbered as in `GATE3_implementation_plan.md` §4 L3. Scenario 4 (citations)
lands with ``source.cited_papers_in_registry``.
"""

from __future__ import annotations

from dataclasses import dataclass


def _manuscript(results: str) -> str:
    """A paper the scanner can read: one section it skips, one it scans."""
    return (
        "\\section{Introduction}\n"
        "We compare GCN and SGC for node classification.\n\n"
        "\\section{Results}\n"
        f"{results}\n"
    )


#: Every number cited through a token.
CLEAN = _manuscript(
    "SGC reaches a test accuracy of \\result{exp1.acc} and trains "
    "\\result{exp2.speedup} times faster than GCN "
    "(\\result{exp2.sgc.wallclock_s} s against \\result{exp2.gcn.wallclock_s} s)."
)

#: The accuracy typed where a token belongs.
TYPED = _manuscript(
    "SGC reaches a test accuracy of 0.812 and trains \\result{exp2.speedup} "
    "times faster than GCN."
)

#: An F1 nobody recorded: the clean run's registry has no ``exp1.f1``.
UNKNOWN = _manuscript(
    "SGC reaches a test accuracy of \\result{exp1.acc} and an F1 of "
    "\\result{exp1.f1}."
)

#: Neither typed nor cited: the Results section states nothing measured.
NUMBERLESS = _manuscript("SGC is much faster than GCN and reaches strong accuracy.")


@dataclass(frozen=True)
class Turn:
    """One writer turn: the manuscript it submitted, and what Gate 3 must do."""

    label: str
    manuscript: str
    #: Check ids that must appear among the blocking failures.
    expect_fail: tuple[str, ...] = ()
    #: Text the feedback sent back for this turn must contain.
    expect_feedback: tuple[str, ...] = ()
    expect_pass: bool = False


@dataclass(frozen=True)
class Scenario:
    name: str
    summary: str
    turns: tuple[Turn, ...]
    #: "pass" - a manuscript was admitted. "raised" - the budget was spent, so
    #: no manuscript is emitted. "no_pass" - the writer stopped first.
    expect_outcome: str
    expect_turns: int
    max_attempts: int = 3


CLEAN_RUN = Scenario(
    name="clean",
    summary="Every number is a result token. Admitted first time.",
    turns=(Turn("tokens only", CLEAN, expect_pass=True),),
    expect_outcome="pass",
    expect_turns=1,
)

TYPED_LITERAL_FIXED = Scenario(
    name="typed-literal-fixed",
    summary="An accuracy typed into Results is rejected; the revision cites the token.",
    turns=(
        Turn(
            "accuracy typed as 0.812",
            TYPED,
            expect_fail=("report.no_numeric_literals_in_results",),
            expect_feedback=("0.812",),
        ),
        Turn("accuracy cited as \\result{exp1.acc}", CLEAN, expect_pass=True),
    ),
    expect_outcome="pass",
    expect_turns=2,
)

UNKNOWN_TOKEN = Scenario(
    name="unknown-token",
    summary="A token naming an F1 nobody recorded is rejected, and the feedback names it.",
    turns=(
        Turn(
            "cites \\result{exp1.f1}",
            UNKNOWN,
            expect_fail=("report.all_tokens_resolve",),
            expect_feedback=("exp1.f1",),
        ),
        Turn("F1 claim dropped", CLEAN, expect_pass=True),
    ),
    expect_outcome="pass",
    expect_turns=2,
)

NO_NUMBERS_IN_RESULTS = Scenario(
    name="no-numbers-in-results",
    summary="A Results section that cites nothing measured is rejected; the revision cites tokens.",
    turns=(
        Turn(
            "results described without a number",
            NUMBERLESS,
            expect_fail=("style.claim_sections_bound",),
            expect_feedback=("results: no \\result{} token",),
        ),
        Turn("results cited with tokens", CLEAN, expect_pass=True),
    ),
    expect_outcome="pass",
    expect_turns=2,
)

BUDGET_EXHAUSTS = Scenario(
    name="budget-exhausts",
    summary="The writer types the accuracy on every turn. Raises; no manuscript.",
    # One turn more than the budget, so the budget ends the loop, not the script.
    turns=tuple(
        Turn(
            f"accuracy typed as 0.812, turn {n}",
            TYPED,
            expect_fail=("report.no_numeric_literals_in_results",),
        )
        for n in range(1, 5)
    ),
    expect_outcome="raised",
    expect_turns=3,
)

SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in (
        BUDGET_EXHAUSTS,
        CLEAN_RUN,
        TYPED_LITERAL_FIXED,
        UNKNOWN_TOKEN,
        NO_NUMBERS_IN_RESULTS,
    )
}
