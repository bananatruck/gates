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

from gates.errors import GateError
from gates.gate3 import (
    GATE_NAME,
    Gate3Config,
    render_result_tokens,
    run_gate3,
)
from gates.report import render_feedback
from gates.schema import Severity, Verdict

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


# --------------------------------------------------------------------------- #
# the measurement
# --------------------------------------------------------------------------- #


ARCHIVED = (
    REPO
    / "reports/finalized-report-and-results/verification/papers/gated"
    / "generated_readme.md"
)


@pytest.mark.skipif(not ARCHIVED.exists(), reason="archived paper not present")
def test_the_archived_manuscript_types_its_own_numbers(tmp_path):
    """Gate 3's first real result, and it is about our own gated arm.

    Gate 1 gated the *execution* of that run. Nothing gated the writing, so the
    model typed the digits directly. Gate 3 rejects it, which is the correct
    outcome and the reason the writing phase needs a gate of its own.
    """
    report = run_gate3(ARCHIVED.read_text(), registry(RECORDED), config(tmp_path))
    literals = check(report, "report.no_numeric_literals_in_results")
    assert report.verdict is Verdict.FAIL
    assert literals.evidence["sections_scanned"] == ["key results"]
    assert len(literals.evidence["literals"]) == 8
