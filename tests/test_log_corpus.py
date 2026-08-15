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


# --------------------------------------------------------------------------- #
# the model tier, measured
# --------------------------------------------------------------------------- #

#: Measured 2026-08-15 against a local qwen3:8b through the shipped prompt,
#: 68 corpus lines per variant, ~7.6s per line. Recorded here so a later run
#: has something to regress against; not asserted by the suite, which must not
#: require a model server.
MODEL_TIER_MEASURED = {
    "deterministic":        {"precision": 1.000, "recall": 0.529, "f1": 0.692},
    "qwen3_8b_no_fewshot":  {"precision": 0.938, "recall": 0.882, "f1": 0.909,
                             "spurious": ["arch-009", "arch-014"]},
    "qwen3_8b_fewshot_k3":  {"precision": 1.000, "recall": 0.882, "f1": 0.938,
                             "spurious": []},
}


def test_the_recorded_model_tier_result_clears_the_precision_floor():
    """The measurement the layer's log-scanning claim rests on.

    Without retrieval the model traded precision for recall and flagged two
    framework-noise lines, one of them the cuDNN registration notice the corpus
    was built around. Balanced few-shot retrieval removed both while holding
    recall, which is what the hard negatives in the exemplar bank are for.
    """
    base = MODEL_TIER_MEASURED["deterministic"]
    naive = MODEL_TIER_MEASURED["qwen3_8b_no_fewshot"]
    best = MODEL_TIER_MEASURED["qwen3_8b_fewshot_k3"]

    assert naive["precision"] < base["precision"]
    assert naive["spurious"] == ["arch-009", "arch-014"]

    assert best["precision"] == base["precision"] == 1.0
    assert best["spurious"] == []
    assert best["recall"] > base["recall"] + 0.3
    assert best["f1"] > naive["f1"] > base["f1"]


def test_the_lines_the_naive_variant_flagged_are_labelled_negative():
    """Both false positives are hard negatives, not corpus mistakes."""
    by_id = {e.id: e for e in CORPUS}
    for spurious in MODEL_TIER_MEASURED["qwen3_8b_no_fewshot"]["spurious"]:
        assert not by_id[spurious].error_signal
    assert by_id["arch-009"].is_hard_negative
