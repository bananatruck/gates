"""Collapsing a log to its distinct shapes.

The property that makes this safe is that it is lossless in distinct content:
every shape survives with a real line number, so nothing a scanner could have
flagged disappears. Recall is not traded for the cost saving — that would be
buying the wrong thing, since recall is exactly what the model tier exists to
raise.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gates.log_digest import DigestLine, compression, digest, shape  # noqa: E402
from rig.corpus import load_corpus  # noqa: E402


def rows(*lines, stream="stdout"):
    return [(stream, i, line) for i, line in enumerate(lines, start=1)]


# --------------------------------------------------------------------------- #
# shaping
# --------------------------------------------------------------------------- #


def test_epoch_lines_share_a_shape():
    a = "epoch 000  loss 1.9243  train_acc 0.4120"
    b = "epoch 199  loss 0.3080  train_acc 0.4120"
    assert shape(a) == shape(b)


def test_nan_survives_as_its_own_shape():
    """The single most important line in a diverged run must not be absorbed."""
    normal = "epoch 012  loss 1.9243  train_acc 0.4120"
    diverged = "epoch 012  loss nan  train_acc 0.4120"
    assert shape(normal) != shape(diverged)


def test_inf_survives_as_its_own_shape():
    assert shape("gradient norm: 12.4") != shape("gradient norm: inf")


def test_timestamps_are_normalised_before_numbers_shred_them():
    a = "2026-07-22 21:45:23.509775: I tensorflow/core/util/port.cc:153] oneDNN on"
    b = "2026-08-15 03:11:02.000001: I tensorflow/core/util/port.cc:153] oneDNN on"
    assert shape(a) == shape(b)


def test_addresses_and_hashes_are_normalised():
    assert shape("at 0x7f96ee9e5a80") == shape("at 0x7fec10df07c0")
    assert shape("run 39f39c717a371a30") == shape("run aeab7fdbffd82055")


def test_different_messages_keep_different_shapes():
    assert shape("Error loading cora: cannot be accessed") != shape(
        "Skipping 3 of 10 folds"
    )


# --------------------------------------------------------------------------- #
# collapsing
# --------------------------------------------------------------------------- #


def test_a_repetitive_loop_becomes_one_row():
    epochs = [f"epoch {i:03d}  loss {1.9 - i * 0.01:.4f}" for i in range(200)]
    digested = digest(rows(*epochs))
    assert len(digested) == 1
    assert digested[0].count == 200
    assert digested[0].lineno == 1


def test_the_first_real_line_number_is_kept_so_grounding_still_resolves():
    digested = digest(rows("setup", "epoch 1 loss 1.0", "epoch 2 loss 0.9", "done"))
    epoch_row = next(d for d in digested if d.count == 2)
    assert epoch_row.lineno == 2


def test_drift_is_preserved_as_a_variant():
    """A run whose accuracy collapses has two lines worth seeing, not one."""
    lines = [f"epoch {i}  acc {0.41 - i * 0.04:.2f}" for i in range(10)]
    digested = digest(rows(*lines))
    assert digested[0].variant is not None
    assert "0.05" in digested[0].variant
    assert digested[0].variant_lineno == 10


def test_identical_lines_produce_no_variant():
    digested = digest(rows("same", "same", "same"))
    assert digested[0].count == 3
    assert digested[0].variant is None


def test_order_follows_first_appearance_not_frequency():
    """A log reads chronologically; that is how setup is told from failure."""
    lines = ["setup done"] + [f"epoch {i}" for i in range(50)] + ["crash imminent"]
    digested = digest(rows(*lines))
    assert [d.line for d in digested] == ["setup done", "epoch 0", "crash imminent"]


def test_streams_do_not_merge():
    candidates = rows("same line", stream="stdout") + rows("same line", stream="stderr")
    digested = digest(candidates)
    assert {d.stream for d in digested} == {"stdout", "stderr"}
    assert all(d.count == 1 for d in digested)


# --------------------------------------------------------------------------- #
# the safety property
# --------------------------------------------------------------------------- #


def test_every_distinct_corpus_line_survives_collapsing():
    """The whole labelled corpus, collapsed, loses no distinct entry.

    If collapsing ever merged two corpus lines, one of them could no longer be
    flagged, and the measured recall would be capped by the digest rather than
    by the model.
    """
    corpus = load_corpus()
    candidates = [(e.stream, i, e.line) for i, e in enumerate(corpus, start=1)]
    digested = digest(candidates)
    assert len(digested) == len(corpus)


def test_a_signal_hidden_among_repetition_still_reaches_the_reader():
    noise = [f"epoch {i}  loss 1.0" for i in range(300)]
    lines = noise[:150] + ["Skipping 3 of 10 folds that raised"] + noise[150:]
    digested = digest(rows(*lines))
    assert any("Skipping 3 of 10 folds" in d.line for d in digested)
    assert len(digested) == 2


def test_compression_is_reported_not_assumed():
    candidates = rows(*[f"epoch {i}" for i in range(100)])
    assert compression(candidates, digest(candidates)) == pytest.approx(0.99)
    assert compression([], []) == 0.0


def test_render_states_the_collapse_rather_than_hiding_it():
    line = DigestLine("stdout", 4, "epoch 0 loss 1.0", count=200)
    assert "x200" in line.render()
    assert DigestLine("stdout", 4, "one off").render() == "one off"
