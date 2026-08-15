"""The labelled corpus, and the baseline every later scanner is measured against.

`logs.no_error_signals` was tuned for precision and its recall was left
explicitly unbounded. This file turns that from a stated position into a
measured one, and pins the precision so no future scanner can trade it away
quietly.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rig.corpus import (  # noqa: E402
    deterministic_scanner,
    load_corpus,
    score,
    wilson,
)

CORPUS = load_corpus()

#: Measured 2026-08-14 against the shipped pattern set. Precision is a floor and
#: must never fall; recall is the number the LLM layer exists to raise.
BASELINE_PRECISION = 1.0
BASELINE_RECALL = 0.529


# --------------------------------------------------------------------------- #
# the corpus is well formed
# --------------------------------------------------------------------------- #


def test_corpus_has_both_classes_in_useful_proportion():
    signals = [e for e in CORPUS if e.error_signal]
    negatives = [e for e in CORPUS if not e.error_signal]
    assert len(CORPUS) >= 60
    assert len(signals) >= 25
    assert len(negatives) >= 25


def test_corpus_ids_are_unique():
    ids = [e.id for e in CORPUS]
    assert len(ids) == len(set(ids))


def test_corpus_draws_on_the_archived_run():
    """Synthetic lines alone would measure the patterns against their own author."""
    archived = [e for e in CORPUS if e.source.startswith("archived-run")]
    assert len(archived) >= 12
    assert any(e.error_signal for e in archived)
    assert any(not e.error_signal for e in archived)


def test_corpus_carries_the_hard_negatives_precision_was_tuned_against():
    hard = {e.line for e in CORPUS if e.is_hard_negative}
    assert any("Mean Squared Error" in line for line in hard)
    assert any("Converged after" in line for line in hard)
    assert any("test error rate" in line for line in hard)


# --------------------------------------------------------------------------- #
# the labelling agrees with the scanner it describes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "entry", [e for e in CORPUS if e.expect_signal], ids=lambda e: e.id
)
def test_entries_naming_a_signal_are_caught_by_that_signal(entry):
    from gates import log_checks

    findings = log_checks.scan(entry.line, entry.stream)
    assert findings, f"{entry.id} claims signal {entry.expect_signal} but was not flagged"
    assert findings[0].signal == entry.expect_signal


@pytest.mark.parametrize(
    "entry", [e for e in CORPUS if e.is_recall_gap], ids=lambda e: e.id
)
def test_recall_gap_entries_really_are_outside_the_pattern_set(entry):
    """If a pattern starts catching one of these, the corpus label is stale."""
    assert not deterministic_scanner(entry.line, entry.stream)


# --------------------------------------------------------------------------- #
# the baseline
# --------------------------------------------------------------------------- #


def test_deterministic_tier_keeps_perfect_precision():
    """The floor. A scanner that flags noise costs the engineer a rewrite for
    nothing, which is worse than missing the signal entirely."""
    result = score(deterministic_scanner, CORPUS)
    assert result.precision == BASELINE_PRECISION
    assert result.spurious == []


def test_deterministic_tier_recall_matches_the_recorded_baseline():
    result = score(deterministic_scanner, CORPUS)
    assert result.recall == pytest.approx(BASELINE_RECALL, abs=0.01)


def test_the_gap_is_the_thing_the_model_layer_is_for():
    """Every miss is a shape no regex was going to reach, not a pattern bug."""
    result = score(deterministic_scanner, CORPUS)
    gap_ids = {e.id for e in CORPUS if e.is_recall_gap}
    assert set(result.missed) == gap_ids


def test_wilson_interval_stays_inside_zero_one():
    assert wilson(0, 10)[0] == 0.0
    assert wilson(10, 10)[1] == 1.0
    lo, hi = wilson(18, 34)
    assert 0.0 < lo < 0.529 < hi < 1.0


def test_interval_widens_as_the_corpus_shrinks():
    narrow = wilson(50, 100)
    wide = wilson(5, 10)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])
