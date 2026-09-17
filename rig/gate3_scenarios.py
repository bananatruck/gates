"""Gate 3 feedback-loop scenarios: the manuscripts a scripted writer submits.

Each turn is the manuscript the writer submitted after reading the previous
feedback, with its ``\\result{}`` tokens intact. Every token names a key from
Gate 2's ``CLEAN`` table, because the registry these manuscripts are judged
against comes from playing a Gate 2 scenario, ``clean`` unless a scenario says
otherwise, not from a hand-built dict (F12). That scenario's declared
limitations come along with its registry.

Numbered as in `GATE3_implementation_plan.md` §4 L3, all six now played, plus
one each for the checks added after the plan was written.
``retrieved`` stands in for the host's search results and literature review,
which no gate produces, so a scripted list is the honest stand-in for them.
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

#: The clean manuscript, with a place for Gate 2's declared limitations.
WITH_LIMITATIONS = CLEAN + "\n\\section{Discussion}\n\\limitations{}\n"

#: The clean manuscript plus the Discussion a host declaring one requires.
WITH_DISCUSSION = CLEAN + "\n\\section{Discussion}\nThe label budget matters less than expected.\n"

#: A figure reference with no \label to resolve against, and the fix. The
#: reference sits on its own line because prose.SKIP_LINE drops any line
#: holding \ref, and the accuracy above it must stay visible to the scanner.
DANGLING_REF = CLEAN + "\nAccuracy is plotted in Figure \\ref{fig:acc}.\n"
LABELLED_REF = DANGLING_REF + "\\begin{figure}\\label{fig:acc}\\end{figure}\n"

#: A related-work line citing a paper no search returned, and one citing a
#: paper the run did retrieve.
FABRICATED = CLEAN + "\n\\section{Related Work}\nWe follow (arXiv 2501.00001v1).\n"
RETRIEVED_CITED = CLEAN + "\n\\section{Related Work}\nWe follow (arXiv 1902.07153v2).\n"


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
    #: The Gate 2 scenario whose registry and declared limitations this one
    #: writes from.
    gate2: str = "clean"
    #: The paper ids the host retrieved. ``None``: the host does not say, and
    #: citations go unchecked.
    retrieved: tuple[str, ...] | None = None
    #: arXiv ids that resolve to a real paper. ``None``: no resolver is injected
    #: and ``source.identifiers_resolve`` emits nothing. Declared as data rather
    #: than a callable so the rig stays a table of scenarios, and the fake keeps
    #: the loop offline (B2).
    resolvable: tuple[str, ...] | None = None
    #: Sections the host declares (D27). Empty by default: these manuscripts are
    #: two sections long because each one exercises a numeric or citation check,
    #: and holding them to the reference host's eight would fail every scenario
    #: for a reason none of them is about. ``missing-section`` declares its own.
    sections: tuple[str, ...] = ()


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

FABRICATED_CITATION = Scenario(
    name="fabricated-citation",
    summary="A citation no search returned is rejected; the revision cites a retrieved paper.",
    turns=(
        Turn(
            "cites arXiv 2501.00001v1, which nothing retrieved",
            FABRICATED,
            expect_fail=("source.cited_papers_in_registry",),
            expect_feedback=("not retrieved: 2501.00001v1",),
        ),
        Turn("cites arXiv 1902.07153v2 from the search results", RETRIEVED_CITED, expect_pass=True),
    ),
    expect_outcome="pass",
    expect_turns=2,
    retrieved=("2410.21676v4", "1902.07153v2"),
)

UNDECLARED_LIMITATION = Scenario(
    name="undeclared-limitation",
    summary="Gate 2 proceeded on a divergence; a paper that omits it is rejected until it states it.",
    turns=(
        Turn(
            "limitations left out",
            CLEAN,
            expect_fail=("report.limitations_declared",),
            expect_feedback=("no \\limitations{} token",),
        ),
        Turn("\\limitations{} placed in the discussion", WITH_LIMITATIONS, expect_pass=True),
    ),
    expect_outcome="pass",
    expect_turns=2,
    gate2="divergence-exhausts",
)

MISSING_SECTION = Scenario(
    name="missing-section",
    summary="A paper omitting a section the host declared is rejected until it writes one.",
    turns=(
        Turn(
            "no discussion section",
            CLEAN,
            expect_fail=("style.sections_present",),
            expect_feedback=("missing:  discussion",),
        ),
        Turn("discussion written", WITH_DISCUSSION, expect_pass=True),
    ),
    expect_outcome="pass",
    expect_turns=2,
    sections=("introduction", "results", "discussion"),
)

UNRESOLVABLE_CITATION = Scenario(
    name="unresolvable-citation",
    summary="A citation naming no paper that exists is rejected; the revision cites a real one.",
    turns=(
        Turn(
            "cites arXiv 2501.00001v1, which no paper carries",
            FABRICATED,
            expect_fail=("source.identifiers_resolve",),
            expect_feedback=("does not resolve: 2501.00001",),
        ),
        Turn("cites arXiv 1902.07153v2, a real paper", RETRIEVED_CITED, expect_pass=True),
    ),
    expect_outcome="pass",
    expect_turns=2,
    # No `retrieved`, so cited_papers_in_registry emits nothing and this scenario
    # isolates the other half of the citation question: not "did we fetch it" but
    # "does it exist".
    resolvable=("1902.07153",),
)

ORPHAN_REFERENCE = Scenario(
    name="orphan-reference",
    summary="A figure reference with no label is rejected until the label exists.",
    turns=(
        Turn(
            "references fig:acc, which has no label",
            DANGLING_REF,
            expect_fail=("style.no_orphan_references",),
            expect_feedback=("no label: fig:acc",),
        ),
        Turn("figure labelled fig:acc", LABELLED_REF, expect_pass=True),
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
        UNDECLARED_LIMITATION,
        FABRICATED_CITATION,
        MISSING_SECTION,
        ORPHAN_REFERENCE,
        UNRESOLVABLE_CITATION,
    )
}
