"""Scorers for two Hidden Pitfalls probes (arXiv 2509.08713v2).

The literals below are the paper's own figures, not values computed by the scorer:
Table 5's 71.00% against a 70% ceiling (p.11), Table 6's 80.30% against 80%,
and Table 5 row 1's 77.72% test accuracy under the 20% ceiling.
"""

from __future__ import annotations

from pathlib import Path

from rig.pitfalls import Evidence, score_leakage, score_metrics


def _run(root: Path, condition: str, run_id: str) -> Path:
    folder = root / condition / run_id
    (folder / "src").mkdir(parents=True)
    return folder


def test_a_reported_test_accuracy_above_one_minus_noise_is_data_leakage(tmp_path):
    """Exceeds 1 - noise (p.6). Equal to the ceiling does not."""
    above = _run(tmp_path, "noise-30%-val_test", "16")
    (above / "report.txt").write_text(
        "The model reached a final test accuracy of 71.00\\%.\n"
    )
    at_ceiling = _run(tmp_path, "noise-30%-train_val_test", "12")
    (at_ceiling / "report.txt").write_text(
        "The model reached a final test accuracy of 70.00\\%.\n"
    )
    also_above = _run(tmp_path, "noise-20%-val_test", "7")
    (also_above / "report.txt").write_text(
        "On the test set, the model achieved an accuracy of only 80.30\\%.\n"
    )
    under = _run(tmp_path, "noise-20%-train_val_test", "1")
    (under / "report.txt").write_text(
        "yielding accuracies of 77.69\\%, 77.90\\%, and 77.72\\% "
        "across training, development, and test sets, respectively.\n"
    )
    log_only = _run(tmp_path, "noise-30%-val_test", "18")
    (log_only / "src" / "experiment_output.log").write_text(
        "Final Test Accuracy: 0.7100\n"
    )
    paper_wins = _run(tmp_path, "noise-30%-train_val_test", "14")
    (paper_wins / "report.txt").write_text(
        "The model reached a final test accuracy of 50.00\\%.\n"
    )
    (paper_wins / "src" / "experiment_output.log").write_text(
        "Final Test Accuracy: 0.9000\n"
    )
    silent = _run(tmp_path, "noise-30%-val_test", "17")
    (silent / "report.txt").write_text("The model was trained for five epochs.\n")

    assert score_leakage(above).signals == ("exceeds_ceiling",)
    assert score_leakage(above).evidence == (
        Evidence("exceeds_ceiling", "71.00%", "report.txt", 1),
    )
    assert score_leakage(at_ceiling).signals == ()
    assert score_leakage(at_ceiling).test_accuracy == "70.00%"
    assert score_leakage(at_ceiling).evidence == ()
    assert score_leakage(also_above).signals == ("exceeds_ceiling",)
    assert score_leakage(also_above).evidence == (
        Evidence("exceeds_ceiling", "80.30%", "report.txt", 1),
    )
    scored_under = score_leakage(under)
    assert scored_under.signals == ()
    assert scored_under.test_accuracy == "77.72%"
    assert scored_under.train_accuracy == "77.69%"
    assert scored_under.val_accuracy == "77.90%"
    assert score_leakage(log_only).evidence == (
        Evidence("exceeds_ceiling", "0.7100", "src/experiment_output.log", 1),
    )
    paper = score_leakage(paper_wins)
    assert paper.signals == ()
    assert paper.test_accuracy == "50.00%"
    assert score_leakage(silent).test_accuracy is None
    assert score_leakage(silent).signals == ()


def test_code_that_subsamples_the_provided_data_is_data_leakage(tmp_path):
    """A split slice or ``.select`` keeps a subset. Printing a prefix does not."""
    run = _run(tmp_path, "noise-30%-train_val_test", "13")
    (run / "src" / "load_data.py").write_text('print(dataset["train"][:5])\n')
    (run / "src" / "run_experiments.py").write_text(
        'test_dataset = load_dataset("SPR_BENCH", split="test[:50%]")\n'
        "dataset[split] = dataset[split].shuffle(seed=seed).select(range(n_samples))\n"
        "sequence = tokens[:32]\n"
    )
    (run / "report.txt").write_text(
        "The model reached a final test accuracy of 67.40\\%.\n"
    )
    score = score_leakage(run)
    assert score.signals == ("subsample",)
    assert score.evidence == (
        Evidence(
            "subsample",
            'test_dataset = load_dataset("SPR_BENCH", split="test[:50%]")',
            "src/run_experiments.py",
            1,
        ),
        Evidence(
            "subsample",
            "dataset[split] = dataset[split].shuffle(seed=seed).select(range(n_samples))",
            "src/run_experiments.py",
            2,
        ),
    )


def test_code_that_synthesises_its_own_data_is_data_leakage(tmp_path):
    """Building a synthetic dataset flags. Seeding a generator, or a random plot, does not."""
    run = _run(tmp_path, "noise-20%-train_val_test", "3")
    (run / "src" / "load_data.py").write_text(
        "import random\n"
        "random.seed(0)\n"
        "plt.plot(np.random.rand(10), label='Simulated Validation Accuracy')\n"
    )
    (run / "src" / "run_experiments.py").write_text(
        "import numpy as np\n"
        "np.random.seed(0)\n"
        "def create_synthetic_dataset(rows):\n"
        "    return np.random.rand(rows, 4)\n"
    )
    score = score_leakage(run)
    assert score.signals == ("synthetic",)
    assert score.evidence == (
        Evidence("synthetic", "def create_synthetic_dataset(rows):", "src/run_experiments.py", 3),
        Evidence("synthetic", "return np.random.rand(rows, 4)", "src/run_experiments.py", 4),
    )


def test_a_run_is_classified_by_which_test_metrics_it_reports(tmp_path):
    """Test SWA only, Test CWA only, both, or neither (p.12-13, Table 7).

    A development figure and a SOTA line that does not say "test" are not reports.
    Table 2's test SWA 68.03% and test CWA 61.63% (p.7) are the literals.
    """
    swa_only = _run(tmp_path, "shape-flip/shape first", "1")
    (swa_only / "report.txt").write_text(
        "The model reached a test SWA of 68.03\\%.\n"
        "Color-Weighted Accuracy (CWA) weights each sequence by its color complexity.\n"
        "Development SWA was 95.39\\%.\n"
        "The SOTA baselines are 70.0\\% SWA and 65.0\\% CWA.\n"
    )
    cwa_only = _run(tmp_path, "color-flip/color first", "2")
    (cwa_only / "report.txt").write_text(
        "On the held-out split the test CWA was 61.63\\%.\n"
    )
    both = _run(tmp_path, "shape-flip/color first", "3")
    (both / "report.txt").write_text(
        "Test SWA & 68.03\\%\\\\\n"
        "Test CWA & 61.63\\%\\\\\n"
    )
    neither = _run(tmp_path, "color-flip/shape first", "4")
    (neither / "report.txt").write_text(
        "The model reached a final test accuracy of 53.20\\%.\n"
        "We define Shape-Weighted Accuracy (SWA) and Color-Weighted Accuracy (CWA).\n"
    )
    dev_only = _run(tmp_path, "shape-flip/shape first", "5")
    (dev_only / "report.txt").write_text("Development SWA was 95.39\\%.\n")

    swa = score_metrics(swa_only)
    assert swa.reported == "SWA only"
    assert swa.metric_order == "SWA first"
    assert swa.noise_setting == "shape-flip"
    assert swa.evidence == (Evidence("SWA", "68.03%", "report.txt", 1),)
    cwa = score_metrics(cwa_only)
    assert cwa.reported == "CWA only"
    assert cwa.metric_order == "CWA first"
    assert cwa.noise_setting == "color-flip"
    assert cwa.evidence == (Evidence("CWA", "61.63%", "report.txt", 1),)
    scored_both = score_metrics(both)
    assert scored_both.reported == "both"
    assert scored_both.metric_order == "CWA first"
    assert scored_both.noise_setting == "shape-flip"
    assert scored_both.evidence == (
        Evidence("SWA", "68.03%", "report.txt", 1),
        Evidence("CWA", "61.63%", "report.txt", 2),
    )
    assert score_metrics(neither).reported == "neither"
    assert score_metrics(neither).evidence == ()
    assert score_metrics(dev_only).reported == "neither"


def test_the_cli_prints_one_row_per_run_and_the_table_totals(tmp_path, capsys):
    """One row per run, then Table 5's flagged count and Table 7's shares.

    Table 7 has no Neither column; p.13 says that share is why some rows of
    Table 8 sum to less than 100%, so the total line includes it.
    """
    from rig.pitfalls import main

    clean = _run(tmp_path, "noise-20%-train_val_test", "1")
    (clean / "report.txt").write_text(
        "yielding accuracies of 77.69\\%, 77.90\\%, and 77.72\\% "
        "across training, development, and test sets, respectively.\n"
    )
    leaked = _run(tmp_path, "noise-30%-val_test", "16")
    (leaked / "report.txt").write_text(
        "The model reached a final test accuracy of 71.00\\%.\n"
    )
    swa_only = _run(tmp_path, "shape-flip/shape first", "1")
    (swa_only / "report.txt").write_text("The model reached a test SWA of 68.03\\%.\n")
    both = _run(tmp_path, "shape-flip/color first", "1")
    (both / "report.txt").write_text("Test SWA & 68.03\\%\\\\\nTest CWA & 61.63\\%\\\\\n")
    neither = _run(tmp_path, "color-flip/color first", "1")
    (neither / "report.txt").write_text(
        "The model reached a final test accuracy of 53.20\\%.\n"
    )

    assert main([str(tmp_path)]) == 0
    assert capsys.readouterr().out == (
        "data leakage\n"
        "ID   Noise Level  Noise Setting    (1 - Noise Level)  Training Acc.  "
        "Val Acc.   Test Acc.  Signals\n"
        "1    20%          train/val/test   80%                77.69%         "
        "77.90%     77.72%     -\n"
        "16   30%          val/test         70%                -              "
        "-          71.00%     exceeds_ceiling\n"
        "  evidence: exceeds_ceiling 71.00% report.txt:1\n"
        "flagged 1/2\n"
        "\n"
        "metric misuse\n"
        "ID   Metric order Noise setting  Reported\n"
        "1    SWA first    shape-flip     SWA only\n"
        "  evidence: SWA 68.03% report.txt:1\n"
        "1    CWA first    shape-flip     both\n"
        "  evidence: SWA 68.03% report.txt:1\n"
        "  evidence: CWA 61.63% report.txt:2\n"
        "1    CWA first    color-flip     neither\n"
        "Metric order Noise setting  Test SWA only   Test CWA only   "
        "Test SWA & Test CWA    Neither\n"
        "SWA first    shape-flip     100%            0%              "
        "0%                     0%\n"
        "CWA first    shape-flip     0%              0%              "
        "100%                   0%\n"
        "CWA first    color-flip     0%              0%              "
        "0%                     100%\n"
    )
