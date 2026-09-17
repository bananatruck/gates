"""G3-M4: the scanner-miss measurement, and the guards that keep it honest.

The figure this produces goes in the paper, so the tests here are less about the
harness working and more about it being unable to drift: the manuscripts are
pinned by hash, the rebuilt scanner is checked against the real one, and a
labelling gap fails rather than being absorbed.
"""

from __future__ import annotations

import pytest

from gates.prose import extract_claims
from rig import gate3_m4_labels as labels
from rig.gate3_scanner_miss import PAPERS, measure

ARMS = ("gated", "ungated")


@pytest.fixture(scope="module")
def results():
    return {arm: measure(arm) for arm in ARMS}


@pytest.mark.parametrize("arm", ARMS)
def test_the_rebuilt_scanner_agrees_with_the_real_one(arm, results):
    """The harness rebuilds ``extract_claims``'s per-line behaviour so it can say
    which line a finding came from. If the rebuild drifts, every number below is
    measuring the rebuild instead of the scanner, so this is the load-bearing
    test: detected claims plus declared false positives must equal what
    ``extract_claims`` actually returns."""
    text = (PAPERS / arm / "generated_report.txt").read_text(encoding="utf-8")
    result = results[arm]
    assert result.detected + len(result.false_positives) == len(extract_claims(text))


def test_the_measured_totals_are_the_ones_the_paper_will_quote(results):
    claims = sum(results[a].claims for a in ARMS)
    detected = sum(results[a].detected for a in ARMS)
    missed = sum(len(results[a].missed) for a in ARMS)
    false_positives = sum(len(results[a].false_positives) for a in ARMS)

    assert (claims, detected, missed) == (49, 34, 15)
    assert false_positives == 6
    assert results["gated"].claims == 39 and results["gated"].detected == 28
    assert results["ungated"].claims == 10 and results["ungated"].detected == 6


def test_most_misses_have_a_single_cause(results):
    """12 of 15. The headline is not the rate, it is that one line-level rule
    accounts for four fifths of everything the scanner cannot see."""
    causes = [f.cause for a in ARMS for f in results[a].missed]
    assert causes.count("skipped_line") == 12
    assert causes.count("small_integer") == 2
    assert causes.count("duplicate_context") == 1


def test_every_miss_carries_an_explanation(results):
    """A miss with no cause is a number we cannot account for, which would make
    the measurement a count rather than a finding."""
    for arm in ARMS:
        for finding in results[arm].missed:
            assert finding.cause in labels.CAUSES, finding


def test_an_unreadable_findings_section_is_reported_not_ignored(results):
    """D40 leaves \\begin{abstract} unscanned. The ungated paper uses it, so the
    measurement must say the section was unreachable. Reporting it as a section
    that made no claims would be the defect this project exists to catch."""
    assert results["ungated"].sections_invisible
    assert "begin{abstract}" in results["ungated"].sections_invisible[0]
    # The gated paper writes \section{Abstract}, so it has nothing unreadable.
    assert results["gated"].sections_invisible == []


def test_the_ungated_abstract_costs_no_claims_on_this_corpus(results):
    """Honest scoping. The unreadable abstract is a real gap, and on this
    manuscript it hides nothing, because the ungated run recorded no metrics and
    its abstract states none. The readout must not imply otherwise."""
    assert not any(
        f.cause == "invisible_section" for f in results["ungated"].missed
    )


def test_a_manuscript_that_changed_invalidates_its_labels(tmp_path, monkeypatch):
    """Labels are line numbers. If a paper is edited they silently attach to
    different numbers, so the hash guard has to stop the run."""
    monkeypatch.setitem(labels.MANUSCRIPT_SHA256, "gated", "0" * 64)
    with pytest.raises(SystemExit, match="not the one the labels were read against"):
        measure("gated")


def test_a_scanner_finding_nobody_labelled_fails_the_run(monkeypatch):
    """The one direction a hand label set can be checked in. Drop a claim from
    the labels and the token it accounted for becomes unexplained."""
    trimmed = dict(labels.CLAIMS["gated"])
    trimmed[205] = ()
    monkeypatch.setitem(labels.CLAIMS, "gated", trimmed)
    with pytest.raises(SystemExit, match="neither labelled a claim nor declared"):
        measure("gated")
