"""Gate 3's numeric binding, checked against manuscripts written by hand.

By hand for the same reason Gate 2's tests are: Gate 3's contract is the
manuscript *format* plus the registry format, and a test that has to run an
experiment and then a writing agent to reach Gate 3 tests two other things
first.

The exception is the last test, which runs the gate over the real archived
paper. That one is not a unit test, it is the measurement.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from gates.adapters.agentlab import (
    gated_report,
    make_report_context,
    make_review_context,
    report_loop,
)
from gates.errors import GateError, GateFailure
from gates.gate3 import (
    GATE_NAME,
    Gate3Config,
    render_result_tokens,
    run_gate3,
)
from gates.report import render_feedback
from gates.schema import Severity, Verdict
from gates.setup import defaults

REPO = pathlib.Path(__file__).resolve().parents[1]


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


def config(tmp_path, **kwargs) -> Gate3Config:
    return Gate3Config(artifact_root=str(tmp_path), **kwargs)


def check(report, check_id):
    return next((c for c in report.checks if c.id == check_id), None)


RECORDED = {
    "exp1.acc_at_100": (0.89, "ratio"),
    "exp1.acc_at_400": (0.97, "ratio"),
    "exp1.efficiency": (0.9175257731958764, "ratio"),
}

TOKENISED = r"""
\section{Results}
Test accuracy reaches \result{exp1.acc_at_400} with the full label budget and
\result{exp1.acc_at_100} with a quarter of it, an efficiency ratio of
\result{exp1.efficiency}.
"""

TYPED = r"""
\section{Results}
Test accuracy reaches 0.97 with the full label budget and 0.89 with a quarter
of it.
"""

MARKDOWN = """
## Key Results

Accuracy climbs to 0.97 once every label is used.
"""


# --------------------------------------------------------------------------- #
# entry conditions
# --------------------------------------------------------------------------- #


def test_rejected_registry_is_not_checkable(tmp_path):
    """Gate 1's rejection stands, and no manuscript built on it is verifiable."""
    reg = registry(RECORDED, citable=False)
    with pytest.raises(GateError, match="Gate 1 did not pass"):
        run_gate3(TOKENISED, reg, config(tmp_path))


def test_report_and_render_are_written_to_the_attempt_dir(tmp_path):
    report = run_gate3(TOKENISED, registry(RECORDED), config(tmp_path))
    written = pathlib.Path(report.artifact_dir)
    assert json.loads((written / "gate3_report.json").read_text())["gate"] == GATE_NAME
    assert "0.97" in (written / "manuscript.rendered").read_text()


# --------------------------------------------------------------------------- #
# report.no_numeric_literals_in_results
# --------------------------------------------------------------------------- #


def test_a_tokenised_manuscript_passes(tmp_path):
    report = run_gate3(TOKENISED, registry(RECORDED), config(tmp_path))
    assert report.verdict is Verdict.PASS


def test_a_typed_numeral_in_the_results_fails(tmp_path):
    report = run_gate3(TYPED, registry(RECORDED), config(tmp_path))
    assert report.verdict is Verdict.FAIL
    literals = check(report, "report.no_numeric_literals_in_results")
    assert {row["value"] for row in literals.evidence["literals"]} == {0.97, 0.89}


def test_a_markdown_manuscript_is_actually_scanned(tmp_path):
    """The false green the scanner move fixed.

    The scanner matched ``\\section{...}`` only, so against a Markdown paper the
    section never left "preamble", no claims were found, and the audit came back
    clean. Both archived manuscripts are Markdown.
    """
    report = run_gate3(MARKDOWN, registry(RECORDED), config(tmp_path))
    literals = check(report, "report.no_numeric_literals_in_results")
    assert literals.evidence["sections_scanned"] == ["key results"]
    assert report.verdict is Verdict.FAIL


def test_a_number_outside_the_findings_sections_is_not_a_claim(tmp_path):
    """Related work quotes other people's numbers. They are not this run's."""
    paper = "\\section{Related work}\nWu et al. report 81.60\\% on Cora.\n"
    report = run_gate3(paper, registry(RECORDED), config(tmp_path))
    assert check(report, "report.no_numeric_literals_in_results").passed


def test_an_unreadable_structure_is_degraded_not_green(tmp_path):
    """A gate that could not find the results must not report them clean."""
    report = run_gate3("Just a paragraph with 0.97 in it.\n", registry(RECORDED),
                       config(tmp_path))
    literals = check(report, "report.no_numeric_literals_in_results")
    assert literals.severity is Severity.INFO
    assert literals.evidence["degraded"] is True
    assert literals.evidence["sections_scanned"] == []


# --------------------------------------------------------------------------- #
# report.all_tokens_resolve
# --------------------------------------------------------------------------- #


def test_a_token_with_no_recorded_value_fails(tmp_path):
    paper = "\\section{Results}\nAccuracy was \\result{exp1.invented}.\n"
    report = run_gate3(paper, registry(RECORDED), config(tmp_path))
    tokens = check(report, "report.all_tokens_resolve")
    assert tokens.evidence["missing"] == ["exp1.invented"]
    assert report.verdict is Verdict.FAIL


def test_an_unresolvable_token_renders_as_itself(tmp_path):
    """Not as an empty string.

    Substituting nothing would turn a missing measurement into a sentence that
    reads fine and says less than it claims, which is the defect one layer down.
    """
    rendered, subs = render_result_tokens(
        "value: \\result{nope}", registry(RECORDED)["values"]
    )
    assert rendered == "value: \\result{nope}"
    assert subs == []


# --------------------------------------------------------------------------- #
# report.rendered_values_match_registry
# --------------------------------------------------------------------------- #


def test_the_renderer_does_not_round(tmp_path):
    """A renderer that formats makes byte-identity false by design."""
    rendered, _ = render_result_tokens(
        "\\result{exp1.efficiency}", registry(RECORDED)["values"]
    )
    assert rendered == "0.9175257731958764"


def test_a_hand_edited_rendered_value_is_caught(tmp_path):
    """The host rendered, then something changed a digit."""
    tampered = "\\section{Results}\nAccuracy reaches 0.99.\n"
    report = run_gate3(
        "\\section{Results}\nAccuracy reaches \\result{exp1.acc_at_400}.\n",
        registry(RECORDED),
        config(tmp_path, rendered=tampered),
    )
    match = check(report, "report.rendered_values_match_registry")
    assert match.passed is False
    assert match.evidence["mismatches"][0]["expected"] == "0.97"


def test_where_the_render_came_from_is_recorded(tmp_path):
    """Self-rendered is the weaker claim, so it is labelled, not assumed."""
    own = run_gate3(TOKENISED, registry(RECORDED), config(tmp_path))
    assert check(own, "report.rendered_values_match_registry").evidence[
        "origin"
    ] == "self_rendered"

    rendered, _ = render_result_tokens(TOKENISED, registry(RECORDED)["values"])
    supplied = run_gate3(
        TOKENISED, registry(RECORDED), config(tmp_path, rendered=rendered)
    )
    assert check(supplied, "report.rendered_values_match_registry").evidence[
        "origin"
    ] == "supplied"


# --------------------------------------------------------------------------- #
# report.figures_referenced_exist
# --------------------------------------------------------------------------- #


def test_a_manuscript_with_no_figure_emits_no_figure_check(tmp_path):
    """A check with nothing to check emits nothing, never a green row."""
    report = run_gate3(TOKENISED, registry(RECORDED), config(tmp_path))
    assert check(report, "report.figures_referenced_exist") is None


def test_a_missing_figure_fails(tmp_path):
    paper = TOKENISED + "\n\\includegraphics[width=0.8]{figures/loss.png}\n"
    report = run_gate3(paper, registry(RECORDED),
                       config(tmp_path, figure_root=str(tmp_path)))
    figures = check(report, "report.figures_referenced_exist")
    assert figures.evidence["missing"][0]["reason"] == "not on disk"


def test_a_present_figure_passes(tmp_path):
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "loss.png").write_bytes(b"\x89PNG")
    paper = TOKENISED + "\n![loss curve](figures/loss.png)\n"
    report = run_gate3(paper, registry(RECORDED),
                       config(tmp_path, figure_root=str(tmp_path)))
    assert check(report, "report.figures_referenced_exist").passed
    assert report.verdict is Verdict.PASS


def test_a_figure_reached_by_escaping_the_run_fails(tmp_path):
    """Whatever it shows, it was not produced by the gated run."""
    # Exists, but one level above the run's artifact root.
    outside = tmp_path / "borrowed.png"
    outside.write_bytes(b"\x89PNG")
    root = tmp_path / "artifacts"
    root.mkdir()
    paper = TOKENISED + "\n\\includegraphics{../borrowed.png}\n"
    report = run_gate3(paper, registry(RECORDED), config(tmp_path, figure_root=str(root)))
    figures = check(report, "report.figures_referenced_exist")
    assert figures.evidence["missing"][0]["reason"] == "outside the run's artifacts"


# --------------------------------------------------------------------------- #
# style.claim_sections_bound
# --------------------------------------------------------------------------- #

NUMBERLESS = r"""
\section{Results}
SGC is much faster than GCN and reaches strong accuracy.

\section{Discussion}
The results suggest SGC is a good default.
"""


def test_a_results_section_that_cites_nothing_fails(tmp_path):
    """The degenerate evasion. Nothing was typed and nothing was cited, so every
    binding check passed while the paper reported no result at all."""
    report = run_gate3(NUMBERLESS, registry(RECORDED), config(tmp_path))
    assert report.verdict is Verdict.FAIL
    bound = check(report, "style.claim_sections_bound")
    assert not bound.passed
    assert bound.evidence["unbound"] == ["results"]


def test_only_a_results_section_must_cite_a_measurement(tmp_path):
    """A discussion with no number in it is honest writing."""
    paper = TOKENISED + "\n\\section{Discussion}\nThe label budget matters less than expected.\n"
    report = run_gate3(paper, registry(RECORDED), config(tmp_path))
    assert report.verdict is Verdict.PASS
    assert check(report, "style.claim_sections_bound").passed


def test_a_subsection_counts_toward_its_results_section(tmp_path):
    paper = (
        "\\section{Results}\nOverview first.\n"
        "\\subsection{Accuracy}\nWe reach \\result{exp1.acc_at_400}.\n"
    )
    report = run_gate3(paper, registry(RECORDED), config(tmp_path))
    assert check(report, "style.claim_sections_bound").passed


def test_a_markdown_results_section_is_held_to_the_same_rule(tmp_path):
    paper = "## Key Results\nAccuracy improves once every label is used.\n"
    report = run_gate3(paper, registry(RECORDED), config(tmp_path))
    assert check(report, "style.claim_sections_bound").evidence["unbound"] == ["key results"]


def test_no_results_section_emits_no_binding_check(tmp_path):
    """Absent, not green. A missing Results heading is style.sections_present's
    to catch, when the host declares its sections (D27)."""
    paper = "\\section{Discussion}\nThe label budget matters less than expected.\n"
    report = run_gate3(paper, registry(RECORDED), config(tmp_path))
    assert check(report, "style.claim_sections_bound") is None


def test_a_manuscript_with_no_numbers_is_not_called_fully_cited(tmp_path):
    """The literal scanner passed NUMBERLESS saying every number came from a
    token. There were no numbers, and a report that says more than was checked
    is the defect this gate exists to catch."""
    literals = check(
        run_gate3(NUMBERLESS, registry(RECORDED), config(tmp_path)),
        "report.no_numeric_literals_in_results",
    )
    assert literals.passed
    assert "every number came from a result token" not in literals.message
    assert "no result token" in literals.message


def test_the_writer_is_told_which_section_and_which_keys(tmp_path):
    text = render_feedback(run_gate3(NUMBERLESS, registry(RECORDED), config(tmp_path)))
    assert "[style.claim_sections_bound]" in text
    assert "results: no \\result{} token" in text
    assert "recorded: exp1.acc_at_100, exp1.acc_at_400, exp1.efficiency" in text
    assert "REQUIRED FIXES" in text


# --------------------------------------------------------------------------- #
# the feedback report
# --------------------------------------------------------------------------- #


def test_feedback_tells_the_writer_what_to_do_instead(tmp_path):
    text = render_feedback(run_gate3(TYPED, registry(RECORDED), config(tmp_path)))
    assert "0.97" in text
    assert "\\result{<key>}" in text
    assert "nothing was run" not in text


def test_feedback_lists_the_keys_that_were_available(tmp_path):
    paper = "\\section{Results}\nAccuracy was \\result{exp1.invented}.\n"
    text = render_feedback(run_gate3(paper, registry(RECORDED), config(tmp_path)))
    assert "missing:  exp1.invented" in text
    assert "exp1.acc_at_400" in text


def test_feedback_lists_every_key_the_writer_may_cite(tmp_path):
    """The writer picks its next token from this line. It used to stop at five
    names without saying so, so a run that recorded six never showed the sixth."""
    six = {f"exp{i}.acc": (0.5, "ratio") for i in range(1, 7)}
    paper = "\\section{Results}\nAccuracy was \\result{exp9.acc}.\n"
    text = render_feedback(run_gate3(paper, registry(six), config(tmp_path)))
    assert "  recorded: " + ", ".join(sorted(six)) + "\n" in text


def test_every_check_gate3_emits_can_be_rendered_and_has_a_fix(tmp_path):
    """D14 for Gate 3, the same guard Gate 2 has. A keyed lookup that misses
    renders nothing, and the writer gets a rejection it cannot act on.

    Every check the fixture emits must also fail, so a check added later that
    this fixture does not trip shows up here instead of passing unguarded."""
    from gates.gate3 import render_result_tokens
    from gates.registry import citable_values
    from gates.report import _EVIDENCE_RENDERERS, _FIXES

    reg = registry(RECORDED)
    paper = (
        "\\section{Results}\nAccuracy was 0.97, or \\result{exp1.acc_at_400} "
        "against \\result{exp1.invented}.\n\\includegraphics{absent.png}\n"
        "\\section{Further Results}\nSGC is fast.\n"
    )
    self_rendered, _ = render_result_tokens(paper, citable_values(reg))
    tampered = self_rendered.replace("0.97", "0.98")
    report = run_gate3(paper, reg, config(tmp_path, figure_root=str(tmp_path), rendered=tampered))
    emitted = {c.id for c in report.checks}

    assert emitted == {c.id for c in report.failed_checks()}
    assert sorted(i for i in emitted if i not in _EVIDENCE_RENDERERS) == []
    assert sorted(i for i in emitted if i not in _FIXES) == []


# --------------------------------------------------------------------------- #
# the host entry point and the loop
# --------------------------------------------------------------------------- #


def report_context(tmp_path, **kwargs):
    return make_report_context(research_dir=str(tmp_path), **kwargs)


def writer(*drafts):
    """A scripted writer: each draft in turn, then stop. Keeps what it was sent."""
    queue = list(drafts)

    def write(feedback):
        write.sent.append(feedback)
        return queue.pop(0) if queue else None

    write.sent = []
    return write


def test_the_host_wiring_path_actually_reaches_gate_3(tmp_path):
    written = gated_report(TOKENISED, registry(RECORDED), report_context(tmp_path))
    assert written.report.gate == GATE_NAME
    assert written.passed


def test_the_report_budget_defaults_to_the_setup_default(tmp_path):
    assert report_context(tmp_path).config.max_attempts == defaults().gate3


def test_a_passing_manuscript_comes_back_rendered(tmp_path):
    outcome = report_loop(report_context(tmp_path), writer(TOKENISED),
                          registry=registry(RECORDED))
    assert outcome.outcome == "pass"
    assert "0.9175257731958764" in outcome.manuscript
    assert "\\result{" not in outcome.manuscript


def test_a_rejected_manuscript_goes_back_to_the_writer(tmp_path):
    write = writer(TYPED, TOKENISED)
    outcome = report_loop(report_context(tmp_path), write, registry=registry(RECORDED))
    assert [w.passed for w in outcome.reports] == [False, True]
    assert write.sent[0] is None
    assert "\\result{<key>}" in write.sent[1]
    assert outcome.outcome == "pass"


def test_a_spent_budget_raises_and_emits_nothing(tmp_path):
    """The opposite of Gate 2: an unverifiable manuscript is not emitted."""
    write = writer(TYPED, TYPED, TOKENISED)
    with pytest.raises(GateFailure) as raised:
        report_loop(report_context(tmp_path, max_attempts=2), write,
                    registry=registry(RECORDED))
    assert raised.value.gate == GATE_NAME
    assert len(write.sent) == 2


def test_a_writer_that_stops_gets_no_manuscript(tmp_path):
    outcome = report_loop(report_context(tmp_path), writer(TYPED),
                          registry=registry(RECORDED))
    assert outcome.outcome == "no_pass"
    assert outcome.manuscript is None
    assert len(outcome.reports) == 1


def test_a_registry_gate_1_rejected_never_reaches_the_writer(tmp_path):
    write = writer(TOKENISED)
    with pytest.raises(GateError, match="Gate 1 did not pass"):
        report_loop(report_context(tmp_path), write,
                    registry=registry(RECORDED, citable=False))
    assert write.sent == []


def test_every_attempt_lands_in_the_ledger(tmp_path):
    context = report_context(tmp_path)
    report_loop(context, writer(TYPED, TOKENISED), registry=registry(RECORDED))
    rows = [r for r in context.ledger.rows() if r["gate"] == GATE_NAME]
    assert [(r["turn"], r["verdict"]) for r in rows] == [(0, "FAIL"), (1, "PASS")]
    assert {r["phase"] for r in rows} == {"report writing"}


def test_gate_3_turns_are_not_counted_as_gate_2_reviews(tmp_path):
    """Both phases append to one ledger, and M5 is Gate 2's number."""
    context = report_context(tmp_path)
    assert make_review_context(research_dir=str(tmp_path)).ledger.path == context.ledger.path
    report_loop(context, writer(TYPED, TOKENISED), registry=registry(RECORDED))
    assert context.ledger.loop_summary()["runs_reviewed"] == 0


# --------------------------------------------------------------------------- #
# the measurement
# --------------------------------------------------------------------------- #


ARCHIVED = (
    REPO
    / "reports/finalized-report-and-results/verification/papers/gated"
    / "generated_report.txt"
)


@pytest.mark.skipif(not ARCHIVED.exists(), reason="archived paper not present")
def test_the_archived_manuscript_types_its_own_numbers(tmp_path):
    """Gate 3's first real result, and it is about our own gated arm.

    Gate 1 gated the *execution* of that run. Nothing gated the writing, so the
    model typed the digits directly. Gate 3 rejects it, which is the correct
    outcome and the reason the writing phase needs a gate of its own.

    The subject is the manuscript itself, LaTeX despite the ``.txt`` suffix.
    This test once read ``generated_readme.md``, a summary the scanner sees as
    one section holding 8 literals, which understated the result (B4).
    """
    report = run_gate3(ARCHIVED.read_text(), registry(RECORDED), config(tmp_path))
    literals = check(report, "report.no_numeric_literals_in_results")
    assert report.verdict is Verdict.FAIL
    assert literals.evidence["sections_scanned"] == [
        "abstract", "results", "discussion"
    ]
    assert len(literals.evidence["literals"]) == 29
