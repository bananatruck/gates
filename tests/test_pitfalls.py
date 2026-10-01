"""Scorers for two Hidden Pitfalls probes (arXiv 2509.08713v2).

The literals below are the paper's own figures, not values computed by the scorer:
Table 5's 71.00% against a 70% ceiling (p.11), Table 6's 80.30% against 80%,
and Table 5 row 1's 77.72% test accuracy under the 20% ceiling.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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


_SUBSTITUTION = "data substitution (our addition, not the paper's leakage)"


def _code_run(root: Path, line: str, run_id: str = "1") -> Path:
    run = _run(root, "noise-30%-train_val_test", run_id)
    (run / "src" / "run_experiments.py").write_text(line + "\n")
    return run


def test_70_009_percent_is_rounded_not_truncated_before_the_strict_comparison(tmp_path):
    """70.009% reads as 70.01 and exceeds 70%. 70.004% reads as 70.00 and does not."""
    above = _run(tmp_path, "noise-30%-val_test", "9")
    (above / "report.txt").write_text("final test accuracy of 70.009\\%.\n")
    level = _run(tmp_path, "noise-30%-val_test", "10")
    (level / "report.txt").write_text("final test accuracy of 70.004\\%.\n")
    assert score_leakage(above).signals == ("exceeds_ceiling",)
    assert score_leakage(level).signals == ()


def test_subsampling_is_data_substitution_and_is_not_counted_as_leakage(tmp_path):
    """p.12: subsampling "does not constitute the data leakage we defined"."""
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
    assert score.signals == ()
    assert score.substitution == ("subsample",)
    assert score.evidence == (
        Evidence(
            _SUBSTITUTION,
            'test_dataset = load_dataset("SPR_BENCH", split="test[:50%]")',
            "src/run_experiments.py",
            1,
            "subsample",
        ),
        Evidence(
            _SUBSTITUTION,
            "dataset[split] = dataset[split].shuffle(seed=seed).select(range(n_samples))",
            "src/run_experiments.py",
            2,
            "subsample",
        ),
    )


_HONEST = [
    'train_ds = load_dataset("csv", data_files=f, split="train[:90%]")',
    'val_ds = load_dataset("csv", data_files=f, split="train[90%:]")',
    "df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)",
    "idx = np.random.permutation(len(X_train))",
    "X_train = X_train[np.random.permutation(len(X_train))]",
    "X_train = np.random.permutation(X_train)",
    "np.random.shuffle(indices)",
    "W1 = np.random.randn(d_in, h) * 0.01",
    "rng = np.random.default_rng(0)",
    "mask = np.random.rand(*h.shape) > p",
    "x = x.select(1, 0)",
    "X_train, X_val = train_test_split(X, test_size=0.1)",
    "tok = random.choice(vocab)",
    "X_train = X_train + np.random.randn(*X_train.shape) * 0.01",
    "sequence = tokens[:32]",
    "logits = out[:, :10]",
    "print(df.head())",
    "print(test_df.head(5))",
    "plt.plot(np.random.rand(10))",
    "fig, axes = plt.subplots(nrows=2, ncols=1)",
    "X_train = X_train[:int(0.9 * n)]",
    "n = len(X_test[:])",
]


@pytest.mark.parametrize("line", _HONEST)
def test_honest_code_is_not_flagged_as_substitution(tmp_path, line):
    score = score_leakage(_code_run(tmp_path, line))
    assert score.substitution == ()
    assert score.evidence == ()


_SUBSAMPLES = [
    "test_df = test_df.head(1000)",
    "test_df = test_df.iloc[:2000]",
    "X_test = X_test[:500]",
    'test = pd.read_csv(p, nrows=1000)',
    "df = df.sample(frac=0.1)",
    "df = df.sample(n=500, random_state=0)",
    "ds = ds.select(range(100))",
    'dev = load_dataset("csv", data_files=f, split="dev[:10%]")',
]


@pytest.mark.parametrize("line", _SUBSAMPLES)
def test_subsampling_of_provided_data_is_caught(tmp_path, line):
    score = score_leakage(_code_run(tmp_path, line))
    assert score.substitution == ("subsample",)
    assert score.signals == ()
    assert [item.line for item in score.evidence] == [1]


_SYNTHETIC = [
    "X = np.random.randn(1000, 5)",
    "y = np.random.randint(0, 2, size=1000)",
    "X_train, y_train = make_classification(n_samples=500)",
    "X = rng.normal(size=(100, 4))",
    "X_test = torch.rand(100, 8)",
    "data = [(random.random(), 1) for _ in range(100)]",
    "def create_synthetic_dataset(rows):",
]


@pytest.mark.parametrize("line", _SYNTHETIC)
def test_a_program_that_makes_its_own_dataset_is_synthetic_data(tmp_path, line):
    """The rule: a data-named variable (X, y, ``*_train``, ``*_test``, data, df ...)
    is assigned straight from a random generator or a ``make_*`` helper."""
    score = score_leakage(_code_run(tmp_path, line))
    assert score.substitution == ("synthetic",)
    assert score.signals == ()
    assert [item.line for item in score.evidence] == [1]


def test_seeding_a_generator_or_a_random_plot_is_not_synthetic_data(tmp_path):
    run = _run(tmp_path, "noise-20%-train_val_test", "3")
    (run / "src" / "load_data.py").write_text(
        "import random\n"
        "random.seed(0)\n"
        "plt.plot(np.random.rand(10), label='Simulated Validation Accuracy')\n"
    )
    assert score_leakage(run).substitution == ()


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


def _metrics(tmp_path, text, order="shape first"):
    run = _run(tmp_path, f"shape-flip/{order}", str(len(list(tmp_path.rglob("report.txt")))))
    (run / "report.txt").write_text(text)
    return score_metrics(run)


def test_a_metric_named_without_a_number_after_it_is_not_reported(tmp_path):
    """Table 3 is a table number. A number before the metric belongs to something else."""
    assert _metrics(tmp_path, "We report test SWA and CWA in Table 3.\n").reported == "neither"
    assert _metrics(tmp_path, "After 3 epochs, the test SWA improved.\n").reported == "neither"
    assert _metrics(tmp_path, "After 3.5 epochs, the test SWA improved.\n").reported == "neither"
    assert _metrics(tmp_path, "At 68.03\\% the test SWA is flat.\n").reported == "neither"


def test_a_sota_or_baseline_clause_is_not_the_runs_own_report(tmp_path):
    own = _metrics(
        tmp_path, "Our test SWA is 68.03\\%, below the SOTA test CWA of 65.0\\%.\n"
    )
    assert own.reported == "SWA only"
    assert own.evidence == (Evidence("SWA", "68.03%", "report.txt", 1),)
    mixed = _metrics(
        tmp_path,
        "Our test SWA is 68.03\\%, compared with SOTA values of 70.0\\% SWA "
        "and 65.0\\% CWA on the test set.\n",
    )
    assert mixed.reported == "SWA only"
    only_sota = _metrics(tmp_path, "SOTA test CWA of 65.0\\%.\n")
    assert only_sota.reported == "neither"
    assert _metrics(tmp_path, "The baseline test SWA was 70.0\\%.\n").reported == "neither"


def test_test_carries_across_clauses_of_one_sentence(tmp_path):
    """A sentence that names the test set once reports every metric figure it gives."""
    listed = _metrics(tmp_path, "Final test SWA = 71.20\\%, CWA = 69.80\\%.\n")
    assert listed.reported == "both"
    assert listed.evidence == (
        Evidence("SWA", "71.20%", "report.txt", 1), Evidence("CWA", "69.80%", "report.txt", 1),
    )
    fronted = _metrics(tmp_path, "On the test set, SWA reached 71.2\\% and CWA 69.8\\%.\n")
    assert fronted.reported == "both"


def test_a_clause_naming_another_split_does_not_inherit_test(tmp_path):
    split = _metrics(tmp_path, "Train SWA is 80.0\\%, test SWA is 71.0\\%.\n")
    assert split.evidence == (Evidence("SWA", "71.0%", "report.txt", 1),)
    assert _metrics(tmp_path, "The test set is held out, validation SWA is 75.0\\%.\n").reported == "neither"


@pytest.mark.parametrize(("text", "expected"), [
    ("Test performance: SWA 71\\%, CWA 69\\% (val: SWA 75\\%, CWA 74\\%).\n",
     (("SWA", "71%"), ("CWA", "69%"))),
    ("Trained on the training set, reaching SWA 90\\%; on test, SWA 71\\%.\n", (("SWA", "71%"),)),
    ("We held out a test set, and validation SWA peaked at 75\\%, CWA at 72\\%.\n", ()),
    ("We did not evaluate on test; validation SWA 75\\%, CWA 70\\%.\n", ()),
    ("Our model reached SWA 70.1\\% (test) and CWA 68.2\\% (validation).\n", (("SWA", "70.1%"),)),
    ("Validation CWA reached 75\\% before the test.\n", ()),
    ("Test SWA 71.0 (val 75.0), CWA 69.0.\n", (("SWA", "71.0"), ("CWA", "69.0"))),
    ("Train/test SWA: 95\\%/71\\%.\n", ()),
    ("Test SWA 71\\%, and the val/test gap in CWA is 5\\%.\n", (("SWA", "71%"),)),
    ("Testing SWA was 71\\%.\n", ()),
    ("Testing hyperparameters, CWA hit 75\\%.\n", ()),
    ("Test SWA is 71\\%, and CWA (validation, 3 seeds) is 75\\%.\n", (("SWA", "71%"),)),
    ("Splits are train, val, test, etc. Our CWA is 75\\%.\n", ()),
    ("SWA (test) 71\\%, CWA (val) 74\\%.\n", (("SWA", "71%"),)),
    ("Test SWA 71\\% vs. 65\\% for prior work, CWA 69\\%.\n", (("SWA", "71%"), ("CWA", "69%"))),
])
def test_a_figure_belongs_to_the_split_named_nearest_before_it(tmp_path, text, expected):
    """Each figure takes the split named closest before it in its sentence, or a (split) tag right after it."""
    got = tuple((e.signal, e.value) for e in _metrics(tmp_path, text).evidence)
    assert got == expected


def test_the_runs_own_figure_before_a_cited_one_is_kept(tmp_path):
    against = _metrics(tmp_path, "test SWA 71.2\\% vs baseline 65\\%.\n")
    assert against.reported == "SWA only"
    assert against.evidence == (Evidence("SWA", "71.2%", "report.txt", 1),)


def test_a_table_cell_after_the_metric_name_is_a_report(tmp_path):
    assert _metrics(tmp_path, "Test results: SWA 68.03\\%\n").reported == "SWA only"
    assert _metrics(tmp_path, "Test SWA & 68.03\\%\\\\\n").reported == "SWA only"


def test_the_cli_prints_one_row_per_run_and_the_table_totals(tmp_path, capsys):
    """One row per run, then Table 5's flagged count and Table 7's shares.

    The flagged count is the paper's leakage only. Data substitution has its own
    column and its own count.

    Table 7 has no Neither column; p.13 says that share is why some rows of
    Table 8 sum to less than 100%, so the total line includes it.
    """
    from rig.pitfalls import main

    clean = _run(tmp_path, "noise-20%-train_val_test", "1")
    (clean / "report.txt").write_text(
        "yielding accuracies of 77.69\\%, 77.90\\%, and 77.72\\% "
        "across training, development, and test sets, respectively.\n"
    )
    substituted = _run(tmp_path, "noise-20%-train_val_test", "3")
    (substituted / "src" / "load.py").write_text("test = pd.read_csv(p, nrows=1000)\n")
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
        "Val Acc.   Test Acc.  Signals          Data substitution (ours)\n"
        "1    20%          train/val/test   80%                77.69%         "
        "77.90%     77.72%     -                -\n"
        "3    20%          train/val/test   80%                -              "
        "-          -          -                subsample\n"
        f"  evidence: {_SUBSTITUTION} (subsample) test = pd.read_csv(p, nrows=1000) "
        "src/load.py:1\n"
        "16   30%          val/test         70%                -              "
        "-          71.00%     exceeds_ceiling  -\n"
        "  evidence: exceeds_ceiling 71.00% report.txt:1\n"
        "flagged 1/3\n"
        f"data substitution 1/3 ({_SUBSTITUTION.split('(', 1)[1]}\n"
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
