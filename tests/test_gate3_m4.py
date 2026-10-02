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

    # D48 measured 34 of 49 with 6 false positives while a \ref skipped its
    # whole line; D76 masks the reference instead and recovers all 12 of those;
    # D104 gives a repeat on one line its own context (47) and reads the
    # ungated abstract, whose one new number is ln 3, a false positive; the
    # thresholds stated twice on lines 305 and 320 count twice each (13).
    assert (claims, detected, missed) == (49, 47, 2)
    assert false_positives == 13
    assert results["gated"].claims == 39 and results["gated"].detected == 37
    assert results["ungated"].claims == 10 and results["ungated"].detected == 10


def test_no_miss_is_a_skipped_line_any_more(results):
    """D48: 12 of 15 misses came from one line-level rule. D76 removed the rule
    for references and citations, so what is left is the scanner's by design:
    small integers. D104 gave a value repeated on one line its own context."""
    causes = [f.cause for a in ARMS for f in results[a].missed]
    assert causes == ["small_integer", "small_integer"]


def test_every_miss_carries_an_explanation(results):
    """A miss with no cause is a number we cannot account for, which would make
    the measurement a count rather than a finding."""
    for arm in ARMS:
        for finding in results[arm].missed:
            assert finding.cause in labels.CAUSES, finding


def test_the_abstract_environment_is_scanned(results):
    """D104: the ungated paper writes \\begin{abstract}, which D40 left unread.
    It is a findings section now, and on this corpus it adds no claim: its one
    number is ln 3, labelled a false positive."""
    assert "abstract" in results["ungated"].sections_scanned
    assert (19, "1.0986", "mathematical constant: ln 3, arithmetic, not a measurement") in (
        results["ungated"].false_positives
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
