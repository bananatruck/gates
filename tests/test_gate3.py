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
    WRITER_SECTIONS,
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
from gates.prose import claim_sections
from gates.report import render_feedback
from gates.schema import PaperRecord, Severity, Verdict
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
# style.sections_present
# --------------------------------------------------------------------------- #

RESULTS_ONLY = "\\section{Results}\nAccuracy reaches \\result{exp1.acc_at_400}.\n"


def test_a_section_the_host_declared_and_the_paper_omits_fails(tmp_path):
    report = run_gate3(
        RESULTS_ONLY,
        registry(RECORDED),
        config(tmp_path, sections=("abstract", "introduction", "results", "discussion")),
    )
    present = check(report, "style.sections_present")
    assert not present.passed
    assert present.evidence["missing"] == ["abstract", "introduction", "discussion"]
    assert report.verdict is Verdict.FAIL


def test_a_host_that_declares_no_sections_gets_no_section_check(tmp_path):
    """D27: `gates/` holds no default list, so nothing is checked and nothing is
    claimed. A default here would be a preference wearing a check's clothes."""
    report = run_gate3(TOKENISED, registry(RECORDED), config(tmp_path))
    assert check(report, "style.sections_present") is None


def test_a_latex_abstract_environment_counts_as_a_declared_abstract(tmp_path):
    """D40. `prose._heading` does not see `\\begin{abstract}` and is not changed:
    presence is a different question from claim scanning, and altering the
    scanner would restate the published Gate 1 traceability number."""
    paper = (
        "\\begin{abstract}\nWe study SGC.\n\\end{abstract}\n" + RESULTS_ONLY
    )
    report = run_gate3(
        paper, registry(RECORDED), config(tmp_path, sections=("abstract", "results"))
    )
    assert check(report, "style.sections_present").passed
    assert claim_sections(paper) == ["results"]


def test_a_declared_section_is_found_inside_a_longer_heading(tmp_path):
    """A host declaring "results" accepts "Experimental Results". The check asks
    whether the section is there, not whether the writer named it our way."""
    paper = "\\section{Experimental Results}\nWe reach \\result{exp1.acc_at_400}.\n"
    report = run_gate3(paper, registry(RECORDED), config(tmp_path, sections=("results",)))
    assert check(report, "style.sections_present").passed


def test_the_writer_is_told_which_declared_section_is_missing(tmp_path):
    text = render_feedback(
        run_gate3(
            RESULTS_ONLY,
            registry(RECORDED),
            config(tmp_path, sections=("results", "discussion")),
        )
    )
    assert "[style.sections_present]" in text
    assert "missing:  discussion" in text
    assert "REQUIRED FIXES" in text


def test_the_reference_host_declares_its_writers_own_sections(tmp_path):
    """D27: the list is the host's, read from `papersolver.py:352` minus
    "scaffold", which is the document skeleton and not a section."""
    context = make_report_context(research_dir=str(tmp_path))
    assert context.config.sections == WRITER_SECTIONS
    assert "scaffold" not in WRITER_SECTIONS
    assert WRITER_SECTIONS[0] == "abstract"


# --------------------------------------------------------------------------- #
# style.no_orphan_references
# --------------------------------------------------------------------------- #


def test_a_reference_to_a_label_nobody_defined_fails(tmp_path):
    """A dangling \\ref renders as "??" in the PDF, so a reader sees it."""
    paper = RESULTS_ONLY + "Accuracy is shown in Figure \\ref{fig:acc}.\n"
    report = run_gate3(paper, registry(RECORDED), config(tmp_path))
    orphans = check(report, "style.no_orphan_references")
    assert not orphans.passed
    assert orphans.evidence["orphans"] == ["fig:acc"]
    assert report.verdict is Verdict.FAIL


def test_a_manuscript_that_cross_references_nothing_emits_no_reference_check(tmp_path):
    """D33: emitted only when the manuscript cross-references at all."""
    report = run_gate3(TOKENISED, registry(RECORDED), config(tmp_path))
    assert check(report, "style.no_orphan_references") is None


def test_a_reference_with_a_matching_label_passes(tmp_path):
    paper = (
        RESULTS_ONLY
        + "Accuracy is shown in Figure \\ref{fig:acc}.\n"
        + "\\begin{figure}\\label{fig:acc}\\end{figure}\n"
    )
    assert check(
        run_gate3(paper, registry(RECORDED), config(tmp_path)),
        "style.no_orphan_references",
    ).passed


def test_a_label_nothing_references_is_not_an_orphan_reference(tmp_path):
    """An unreferenced float is style.floats_referenced's, at WARN. A label on a
    section nobody points at is nobody's. Neither is a dangling reference."""
    paper = RESULTS_ONLY + "\\begin{figure}\\label{fig:acc}\\end{figure}\n"
    orphans = check(
        run_gate3(paper, registry(RECORDED), config(tmp_path)),
        "style.no_orphan_references",
    )
    assert orphans.passed
    assert "no cross-reference to resolve" in orphans.message


def test_a_cleveref_list_is_split_into_its_targets(tmp_path):
    """\\cref{a,b} is two references. Reading it as one target named "a,b"
    would report an orphan that does not exist and miss the one that does."""
    paper = (
        RESULTS_ONLY
        + "See \\cref{fig:acc,fig:loss}.\n"
        + "\\begin{figure}\\label{fig:acc}\\end{figure}\n"
    )
    orphans = check(
        run_gate3(paper, registry(RECORDED), config(tmp_path)),
        "style.no_orphan_references",
    )
    assert orphans.evidence["orphans"] == ["fig:loss"]


def test_the_writer_is_told_which_reference_has_no_label(tmp_path):
    paper = (
        RESULTS_ONLY
        + "Accuracy is in Figure \\ref{fig:acc}.\n"
        + "\\begin{table}\\label{tab:main}\\end{table}\n"
    )
    text = render_feedback(run_gate3(paper, registry(RECORDED), config(tmp_path)))
    assert "[style.no_orphan_references]" in text
    assert "no label: fig:acc" in text
    assert "defined:  tab:main" in text


# --------------------------------------------------------------------------- #
# source.identifiers_resolve
# --------------------------------------------------------------------------- #

CITES = RESULTS_ONLY + "We follow (arXiv 2410.21676v4).\n"


def fake_lookup(*known: str, fails: bool = False):
    """A dict-backed resolver, the shape B2 keeps the suite hermetic with."""

    def lookup(identifier):
        lookup.asked.append(identifier)
        if fails:
            raise OSError("export.arxiv.org: connection refused")
        if identifier not in known:
            return None
        return PaperRecord(
            identifier=identifier,
            title="Scaling Laws",
            authors=("Kaplan",),
            year=2020,
            locator=f"https://arxiv.org/abs/{identifier}",
            content_hash="deadbeef",
        )

    lookup.asked = []
    return lookup


def test_a_citation_that_resolves_to_a_real_record_passes(tmp_path):
    report = run_gate3(
        CITES, registry(RECORDED), config(tmp_path, lookup=fake_lookup("2410.21676"))
    )
    assert check(report, "source.identifiers_resolve").passed


def test_a_citation_that_resolves_to_nothing_fails(tmp_path):
    """The fabricated-identifier case. source.cited_papers_in_registry catches a
    paper this run never retrieved; this catches one that does not exist."""
    report = run_gate3(CITES, registry(RECORDED), config(tmp_path, lookup=fake_lookup()))
    resolved = check(report, "source.identifiers_resolve")
    assert not resolved.passed
    assert resolved.evidence["unresolved"] == ["2410.21676"]
    assert report.verdict is Verdict.FAIL


def test_without_a_lookup_citations_are_not_resolved_at_all(tmp_path):
    """B2 and the standing rule: no input, no check, and no green row."""
    report = run_gate3(CITES, registry(RECORDED), config(tmp_path))
    assert check(report, "source.identifiers_resolve") is None


def test_a_lookup_that_cannot_reach_the_network_says_so(tmp_path):
    """The plan's S1. When the network is down the check does not pass, and the
    report carries the absence so a reader knows citations went unchecked."""
    report = run_gate3(
        CITES, registry(RECORDED), config(tmp_path, lookup=fake_lookup(fails=True))
    )
    resolved = check(report, "source.identifiers_resolve")
    assert resolved.severity is Severity.INFO
    assert resolved.evidence["degraded"] is True
    assert "could not" in resolved.message
    # An outage is not a defect in the manuscript, so it cannot block it.
    assert report.verdict is Verdict.PASS


def test_a_fabricated_citation_still_fails_when_another_lookup_breaks(tmp_path):
    """Duty 1. An outage must not launder a fabrication: once an identifier is
    known to resolve to nothing, a later resolver failure cannot turn the whole
    check into "could not check" and let the manuscript through."""

    def lookup(identifier):
        if identifier == "2501.00001":
            return None
        raise OSError("export.arxiv.org: connection refused")

    paper = (
        RESULTS_ONLY
        + "We follow (arXiv 2501.00001v1) and (arXiv 2410.21676v4).\n"
    )
    report = run_gate3(paper, registry(RECORDED), config(tmp_path, lookup=lookup))
    resolved = check(report, "source.identifiers_resolve")
    assert resolved.severity is Severity.FAIL
    assert resolved.evidence["unresolved"] == ["2501.00001"]
    assert report.verdict is Verdict.FAIL
    # And the manuscript is still told the rest went unchecked.
    assert resolved.evidence["unchecked"] == ["2410.21676"]


def test_the_lookup_is_asked_for_the_version_stripped_identifier(tmp_path):
    """D26 makes the version-stripped arXiv id canonical. Whether v4 exists is
    source.cited_papers_in_registry's question, against what the run read."""
    lookup = fake_lookup("2410.21676")
    run_gate3(CITES, registry(RECORDED), config(tmp_path, lookup=lookup))
    assert lookup.asked == ["2410.21676"]


def test_a_doi_is_never_sent_to_an_arxiv_lookup(tmp_path):
    """D26: the reference host never sees a DOI, so a cited DOI is already a
    failure in source.cited_papers_in_registry. Resolving it would be asking
    arXiv about an identifier it does not issue."""
    paper = RESULTS_ONLY + "We follow 10.1145/3292500.3330701.\n"
    lookup = fake_lookup()
    run_gate3(paper, registry(RECORDED), config(tmp_path, lookup=lookup))
    assert lookup.asked == []


def test_the_writer_is_told_which_identifier_did_not_resolve(tmp_path):
    text = render_feedback(
        run_gate3(CITES, registry(RECORDED), config(tmp_path, lookup=fake_lookup()))
    )
    assert "[source.identifiers_resolve]" in text
    assert "does not resolve: 2410.21676" in text


def test_a_resolved_record_carries_where_it_came_from(tmp_path):
    """PaperRecord's locator and content_hash exist so "the same paper" is
    checkable later, which is what any future tier B corpus work needs."""
    report = run_gate3(
        CITES, registry(RECORDED), config(tmp_path, lookup=fake_lookup("2410.21676"))
    )
    resolved = check(report, "source.identifiers_resolve")
    assert resolved.evidence["resolved"] == [
        {
            "identifier": "2410.21676",
            "title": "Scaling Laws",
            "year": 2020,
            "locator": "https://arxiv.org/abs/2410.21676",
        }
    ]


# --------------------------------------------------------------------------- #
# style.floats_referenced
# --------------------------------------------------------------------------- #


def test_a_labelled_figure_the_text_never_points_at_warns(tmp_path):
    """WARN, not FAIL (D33). A figure the prose never mentions is a drafting
    slip, and costing the writer a turn for it could cost the paper."""
    paper = RESULTS_ONLY + "\\begin{figure}\\label{fig:acc}\\end{figure}\n"
    report = run_gate3(paper, registry(RECORDED), config(tmp_path))
    floats = check(report, "style.floats_referenced")
    assert floats.severity is Severity.WARN
    assert not floats.passed
    assert floats.evidence["unreferenced"] == [{"label": "fig:acc", "kind": "figure"}]
    # A warning never moves the verdict.
    assert report.verdict is Verdict.PASS


def test_a_labelled_table_is_held_to_the_same_rule(tmp_path):
    paper = RESULTS_ONLY + "\\begin{table}\\label{tab:main}\\end{table}\n"
    floats = check(
        run_gate3(paper, registry(RECORDED), config(tmp_path)),
        "style.floats_referenced",
    )
    assert floats.evidence["unreferenced"] == [{"label": "tab:main", "kind": "table"}]


def test_a_referenced_float_does_not_warn(tmp_path):
    paper = (
        RESULTS_ONLY
        + "\\begin{figure}\\label{fig:acc}\\end{figure}\n"
        + "Accuracy is plotted in Figure \\ref{fig:acc}.\n"
    )
    assert check(
        run_gate3(paper, registry(RECORDED), config(tmp_path)),
        "style.floats_referenced",
    ).passed


def test_a_manuscript_with_no_labelled_float_emits_no_float_check(tmp_path):
    """A label on a section is not a float, so there is nothing to reference."""
    paper = RESULTS_ONLY + "\\label{sec:results}\n"
    report = run_gate3(paper, registry(RECORDED), config(tmp_path))
    assert check(report, "style.floats_referenced") is None


def test_a_float_labelled_on_a_later_line_still_counts(tmp_path):
    """The reference host writes multi-line environments, so the label rarely
    sits on the \\begin line."""
    paper = (
        RESULTS_ONLY
        + "\\begin{figure}\n\\includegraphics{acc.png}\n"
        + "\\caption{Accuracy}\n\\label{fig:acc}\n\\end{figure}\n"
    )
    floats = check(
        run_gate3(paper, registry(RECORDED), config(tmp_path)),
        "style.floats_referenced",
    )
    assert floats.evidence["unreferenced"] == [{"label": "fig:acc", "kind": "figure"}]


def test_the_writer_is_told_which_float_goes_unmentioned(tmp_path):
    paper = RESULTS_ONLY + "\\begin{figure}\\label{fig:acc}\\end{figure}\n"
    text = render_feedback(run_gate3(paper, registry(RECORDED), config(tmp_path)))
    assert "[style.floats_referenced]" in text
    assert "never referenced: fig:acc (figure)" in text


# --------------------------------------------------------------------------- #
# report.limitations_declared
# --------------------------------------------------------------------------- #

#: What Gate 2's review_loop hands over after a spent budget, verbatim.
DECLARED = (
    "DECLARED LIMITATIONS\n\n"
    "  - config.lr: the plan declared 0.001 and the run recorded 0.01 "
    "(plan L4: learning rate 0.001)\n"
)

WITH_LIMITATIONS = TOKENISED + "\n\\section{Discussion}\n\\limitations{}\n"


def test_declared_limitations_are_rendered_word_for_word(tmp_path):
    """D28. Rendered, not authored: the writer places the token and the renderer
    inserts Gate 2's text, so no paraphrase can soften it."""
    report = run_gate3(WITH_LIMITATIONS, registry(RECORDED), config(tmp_path),
                       declared=DECLARED)
    rendered = (pathlib.Path(report.artifact_dir) / "manuscript.rendered").read_text()
    assert report.verdict is Verdict.PASS
    assert check(report, "report.limitations_declared").passed
    assert "\\begin{verbatim}\n" + DECLARED + "\\end{verbatim}" in rendered
    assert "\\limitations{}" not in rendered


def test_a_manuscript_without_the_limitations_token_fails(tmp_path):
    report = run_gate3(TOKENISED, registry(RECORDED), config(tmp_path),
                       declared=DECLARED)
    declared = check(report, "report.limitations_declared")
    assert report.verdict is Verdict.FAIL
    assert declared.evidence["token_found"] is False
    assert declared.evidence["missing"] == [
        "DECLARED LIMITATIONS",
        "- config.lr: the plan declared 0.001 and the run recorded 0.01 "
        "(plan L4: learning rate 0.001)",
    ]


def test_nothing_to_declare_emits_no_limitations_check(tmp_path):
    """Absent, not green. The token is still allowed and renders as nothing."""
    report = run_gate3(WITH_LIMITATIONS, registry(RECORDED), config(tmp_path))
    rendered = (pathlib.Path(report.artifact_dir) / "manuscript.rendered").read_text()
    assert check(report, "report.limitations_declared") is None
    assert "\\limitations{}" not in rendered
    assert report.verdict is Verdict.PASS


def test_a_host_render_that_drops_a_limitation_fails(tmp_path):
    """When the host renders, its text is the one checked."""
    from gates.gate3 import render_result_tokens
    from gates.registry import citable_values

    own, _ = render_result_tokens(WITH_LIMITATIONS, citable_values(registry(RECORDED)),
                                  declared=DECLARED)
    dropped = own.replace("the run recorded 0.01", "the run differed")
    report = run_gate3(WITH_LIMITATIONS, registry(RECORDED),
                       config(tmp_path, rendered=dropped), declared=DECLARED)
    declared = check(report, "report.limitations_declared")
    assert declared.evidence["token_found"] is True
    assert declared.evidence["origin"] == "supplied"
    assert len(declared.evidence["missing"]) == 1
    assert report.verdict is Verdict.FAIL


def test_numbers_inside_a_limitation_are_not_typed_literals(tmp_path):
    """0.001 and 0.01 arrive with Gate 2's text, not from the writer's hand."""
    paper = "\\section{Results}\nWe reach \\result{exp1.acc_at_400}.\n\\limitations{}\n"
    report = run_gate3(paper, registry(RECORDED), config(tmp_path), declared=DECLARED)
    assert check(report, "report.no_numeric_literals_in_results").passed
    assert report.verdict is Verdict.PASS


def test_result_values_placed_after_the_limitations_still_match(tmp_path):
    """The block is inserted before result tokens, so their offsets stay true."""
    paper = "\\section{Results}\n\\limitations{}\nWe reach \\result{exp1.acc_at_400}.\n"
    report = run_gate3(paper, registry(RECORDED), config(tmp_path), declared=DECLARED)
    assert check(report, "report.rendered_values_match_registry").passed


def test_latex_specials_in_a_limitation_cannot_break_or_hide_it(tmp_path):
    """A bare ``_`` stops LaTeX compiling, and a ``%`` comments out the rest of
    its line, so the limitation would sit in the source and never print."""
    from gates.gate3 import render_result_tokens

    special = "DECLARED LIMITATIONS\n\n  - exp2.gcn.wallclock_s: 12% slower than declared\n"
    rendered, _ = render_result_tokens("\\limitations{}", {}, declared=special)
    assert rendered == "\\begin{verbatim}\n" + special + "\\end{verbatim}"


def test_the_writer_is_told_to_place_the_limitations(tmp_path):
    text = render_feedback(run_gate3(TOKENISED, registry(RECORDED), config(tmp_path),
                                     declared=DECLARED))
    assert "[report.limitations_declared]" in text
    assert "no \\limitations{} token in the manuscript" in text
    assert "the run recorded 0.01" in text
    assert "Do not paraphrase" in text


# --------------------------------------------------------------------------- #
# source.cited_papers_in_registry
# --------------------------------------------------------------------------- #

#: What the host's search tool and literature review returned (D25).
RETRIEVED = {"2410.21676v2", "1902.07153v2", "2412.07942v1"}


def cites(*ids: str) -> str:
    return TOKENISED + "\\section{Related Work}\n" + " ".join(f"(arXiv {i})" for i in ids) + "\n"


def test_a_paper_nobody_retrieved_fails(tmp_path):
    """MLR-Bench's incorrect citation: an id no search or review returned."""
    report = run_gate3(cites("1902.07153v2", "2501.00001v1"), registry(RECORDED),
                       config(tmp_path), retrieved=RETRIEVED)
    papers = check(report, "source.cited_papers_in_registry")
    assert report.verdict is Verdict.FAIL
    assert papers.evidence["not_retrieved"] == ["2501.00001v1"]


def test_versions_are_compared_stripped_and_a_mismatch_is_evidence(tmp_path):
    """D26. v4 of a paper the run read as v2 is the same paper."""
    report = run_gate3(cites("2410.21676v4"), registry(RECORDED), config(tmp_path),
                       retrieved=RETRIEVED)
    papers = check(report, "source.cited_papers_in_registry")
    assert papers.passed
    assert papers.evidence["version_mismatches"] == [
        {"cited": "2410.21676v4", "retrieved": ["2410.21676v2"]}
    ]
    assert papers.evidence["discrepancies"] == []


def test_an_unversioned_citation_matches_any_version(tmp_path):
    report = run_gate3(cites("1902.07153"), registry(RECORDED), config(tmp_path),
                       retrieved=RETRIEVED)
    papers = check(report, "source.cited_papers_in_registry")
    assert papers.passed
    assert papers.evidence["version_mismatches"] == []


def test_the_colon_form_is_read_too(tmp_path):
    paper = TOKENISED + "As shown in arXiv:2501.00001, it works.\n"
    report = run_gate3(paper, registry(RECORDED), config(tmp_path), retrieved=RETRIEVED)
    assert check(report, "source.cited_papers_in_registry").evidence["not_retrieved"] == [
        "2501.00001"
    ]


def test_a_cited_doi_fails_as_not_retrieved(tmp_path):
    """D26: the reference host never sees a DOI, so nothing could have retrieved one."""
    paper = TOKENISED + "See doi:10.1145/3292500.3330925 for details.\n"
    report = run_gate3(paper, registry(RECORDED), config(tmp_path), retrieved=RETRIEVED)
    assert check(report, "source.cited_papers_in_registry").evidence["not_retrieved"] == [
        "10.1145/3292500.3330925"
    ]


def test_no_retrieval_record_emits_no_citation_check(tmp_path):
    """Absent, not green: a host that does not say what it retrieved gets no row."""
    report = run_gate3(cites("2501.00001v1"), registry(RECORDED), config(tmp_path))
    assert check(report, "source.cited_papers_in_registry") is None


def test_a_manuscript_citing_nothing_emits_no_citation_check(tmp_path):
    report = run_gate3(TOKENISED, registry(RECORDED), config(tmp_path), retrieved=RETRIEVED)
    assert check(report, "source.cited_papers_in_registry") is None


def test_the_writer_is_told_which_citation_and_what_it_may_cite(tmp_path):
    text = render_feedback(run_gate3(cites("2501.00001v1"), registry(RECORDED),
                                     config(tmp_path), retrieved=RETRIEVED))
    assert "[source.cited_papers_in_registry]" in text
    assert "not retrieved: 2501.00001v1" in text
    assert "retrieved: 1902.07153v2, 2410.21676v2, 2412.07942v1" in text
    assert "treated as fabricated" in text


def test_the_host_retrieval_record_is_read_from_its_own_formats():
    """``lit_review`` entries carry ``arxiv_id`` (D21). The writer's search
    results are text with an ``arXiv paper ID:`` line per paper (D25)."""
    from gates.adapters.agentlab import retrieved_arxiv_ids

    lit_review = [{"arxiv_id": "2412.07942v1", "summary": "..."}, {"summary": "no id"}]
    related = {
        "introduction": "Title: A\nSummary: ...\narXiv paper ID: 2410.21676v4\n\n"
                        "Title: B\narXiv paper ID: 1902.07153v2\n",
        "discussion": None,
    }
    assert retrieved_arxiv_ids(lit_review, related) == {
        "2412.07942v1", "2410.21676v4", "1902.07153v2"
    }


ARCHIVED_LOG = (
    REPO / "reports/finalized-report-and-results/verification/logs/gated_workflow.log"
)


@pytest.mark.skipif(not ARCHIVED_LOG.exists(), reason="archived log not present")
def test_the_archived_run_can_only_be_measured_against_its_literature_review(tmp_path):
    """The measurement, with its limit stated. The archived run logged its
    ADD_PAPER commands but not the writer's per-section search results, so only
    D21's registry can be rebuilt from it. Under that registry 7 of the 8 cited
    papers flag, and D25 exists because most of those came from the writer's
    own searches. The D25 number needs a new run."""
    lines = ARCHIVED_LOG.read_text(encoding="utf-8").splitlines()
    added = {lines[i + 1].strip() for i, line in enumerate(lines) if line.strip() == "```ADD_PAPER"}
    assert added == {"2412.07942v1", "2201.12150v2"}
    assert "arXiv paper ID" not in ARCHIVED_LOG.read_text(encoding="utf-8")

    paper = ARCHIVED.read_text(encoding="utf-8")
    report = run_gate3(paper, registry(RECORDED), config(tmp_path), retrieved=added)
    papers = check(report, "source.cited_papers_in_registry")
    assert len(papers.evidence["cited"]) == 8
    assert len(papers.evidence["not_retrieved"]) == 7


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

    Every check the fixture emits, INFO rows aside, must also fail or warn, so a
    check added later that this fixture does not trip shows up here instead of
    passing unguarded. A scripted model makes the claim scan warn."""
    import json as _json

    from gates import llm_claims
    from gates.gate3 import render_result_tokens
    from gates.registry import citable_values
    from gates.report import _EVIDENCE_RENDERERS, _FIXES

    def model(prompt, system):
        if system != llm_claims.SYSTEM:
            return "1. Cite `exp1.acc_at_400`."
        return _json.dumps([{"line": 2, "quote": "fast", "why": "a speed claim"}])

    reg = registry(RECORDED)
    paper = (
        "\\section{Results}\nAccuracy was 0.97, or \\result{exp1.acc_at_400} "
        "against \\result{exp1.invented}.\n\\includegraphics{absent.png}\n"
        "\\section{Further Results}\nSGC is fast.\n"
        "\\section{Related Work}\nAs in (arXiv 2501.00001v1).\n"
        # Last, and outside the findings sections: the claim scan numbers the
        # rows it hands the model, so a line added above "SGC is fast" would
        # shift the row this fixture's scripted model quotes.
        "See Figure \\ref{fig:absent}.\n"
        "\\begin{figure}\\label{fig:unmentioned}\\end{figure}\n"
    )
    self_rendered, _ = render_result_tokens(paper, citable_values(reg))
    tampered = self_rendered.replace("0.97", "0.98")
    report = run_gate3(
        paper,
        reg,
        config(
            tmp_path,
            figure_root=str(tmp_path),
            rendered=tampered,
            # The fixture paper has no abstract, so this trips sections_present.
            sections=("abstract", "results"),
            # Resolves nothing, so the cited id trips identifiers_resolve.
            lookup=lambda identifier: None,
            consult_model=model,
        ),
        declared="DECLARED LIMITATIONS\n\n  - a.b: unresolved\n",
        retrieved={"1902.07153v2"},
    )
    emitted = {c.id for c in report.checks if c.severity is not Severity.INFO}
    failed = {c.id for c in report.failed_checks()}

    assert emitted == failed | {c.id for c in report.warnings()}
    assert "report.model_unbound_claims" in emitted
    assert sorted(i for i in emitted if i not in _EVIDENCE_RENDERERS) == []
    assert sorted(i for i in failed if i not in _FIXES) == []


# --------------------------------------------------------------------------- #
# the host entry point and the loop
# --------------------------------------------------------------------------- #


def report_context(tmp_path, **kwargs):
    """The loop's context, with sections unchecked unless a test asks for them.

    These tests write two-line manuscripts to exercise the loop, and the real
    adapter default holds a paper to all eight of the host's sections
    (``WRITER_SECTIONS``), which every fixture here would fail. That default is
    pinned by ``test_the_reference_host_declares_its_writers_own_sections``.
    """
    kwargs.setdefault("sections", ())
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


def test_the_loop_holds_the_writer_to_gate_2s_limitations(tmp_path):
    """The declared block travels with the registry, both out of Gate 2's run."""
    write = writer(TOKENISED, WITH_LIMITATIONS)
    outcome = report_loop(report_context(tmp_path), write, registry=registry(RECORDED),
                          declared=DECLARED)
    assert [w.passed for w in outcome.reports] == [False, True]
    assert "\\limitations{}" in write.sent[1]
    assert "the run recorded 0.01" in outcome.manuscript


def test_the_retrieval_record_is_read_after_each_write(tmp_path):
    """D32. The host fills its search results while writing, so a record read
    before the first write would miss every paper the writer just found."""
    found: set[str] = set()

    def write(feedback):
        found.add("2501.00001v1")
        return cites("2501.00001v1")

    outcome = report_loop(report_context(tmp_path), write, registry=registry(RECORDED),
                          retrieved=lambda: found)
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
