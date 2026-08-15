"""Measure a log scanner against the labelled corpus.

`logs.no_error_signals` has recall it has never been able to bound —
GATE1_REQUIREMENTS.md V3 says so in as many words: "an error message outside it
passes silently." That is a defensible position only while the size of the gap is
unknown. This module measures it.

The corpus is `tests/fixtures/log_corpus.jsonl`: one labelled line per record,
`error_signal` true when the line reports trouble the run continued past and that
changes what the surrounding numbers mean. Roughly a quarter of the entries are
taken verbatim from `results/gemini_3_5_flash_run_1/`; the rest cover the
numerical, device and convergence shapes the pattern set was written for, the
hard negatives it was tuned against, and the gap shapes it was never going to
catch.

Precision is the number that governs, though for a reason worth stating
precisely: these findings are WARN and cannot force a rewrite, so a false
positive's real cost is that the writer is told to disclose a problem that never
happened. A scanner that improves recall by flagging TensorFlow's
plugin-registration noise has made the paper worse, not the compute bill.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

CORPUS_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "log_corpus.jsonl"

#: A scanner takes one line and its stream and says whether it is an error
#: signal. Both the deterministic tier and the model tier reduce to this.
ScannerFn = Callable[[str, str], bool]


@dataclass(frozen=True)
class CorpusEntry:
    id: str
    source: str
    stream: str
    line: str
    error_signal: bool
    expect_signal: str | None
    note: str | None

    @property
    def is_recall_gap(self) -> bool:
        """A true signal the deterministic pattern set was never going to catch."""
        return self.error_signal and self.expect_signal is None

    @property
    def is_hard_negative(self) -> bool:
        return bool(self.note and self.note.startswith("HARD NEGATIVE"))


def load_corpus(path: Path | str = CORPUS_PATH) -> list[CorpusEntry]:
    entries: list[CorpusEntry] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            entries.append(
                CorpusEntry(
                    id=row["id"],
                    source=row["source"],
                    stream=row["stream"],
                    line=row["line"],
                    error_signal=row["error_signal"],
                    expect_signal=row.get("expect_signal"),
                    note=row.get("note"),
                )
            )
    return entries


@dataclass
class Score:
    """Confusion matrix and the rates derived from it."""

    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0
    missed: list[str] = None  # type: ignore[assignment]
    spurious: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.missed = self.missed or []
        self.spurious = self.spurious or []

    @property
    def positives(self) -> int:
        return self.true_positive + self.false_negative

    @property
    def flagged(self) -> int:
        return self.true_positive + self.false_positive

    @property
    def precision(self) -> float:
        return self.true_positive / self.flagged if self.flagged else 1.0

    @property
    def recall(self) -> float:
        return self.true_positive / self.positives if self.positives else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def interval(self, rate: str = "recall") -> tuple[float, float]:
        """Wilson score interval at 95%.

        Wilson rather than normal-approximation because the corpus is small and
        the rates sit near the ends, where the normal interval runs past 0 and 1
        and stops meaning anything.
        """
        if rate == "recall":
            successes, trials = self.true_positive, self.positives
        else:
            successes, trials = self.true_positive, self.flagged
        return wilson(successes, trials)

    def to_dict(self) -> dict:
        lo_r, hi_r = self.interval("recall")
        lo_p, hi_p = self.interval("precision")
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "false_negative": self.false_negative,
            "precision": round(self.precision, 4),
            "precision_ci95": [round(lo_p, 4), round(hi_p, 4)],
            "recall": round(self.recall, 4),
            "recall_ci95": [round(lo_r, 4), round(hi_r, 4)],
            "f1": round(self.f1, 4),
            "missed": self.missed,
            "spurious": self.spurious,
        }


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def score(scanner: ScannerFn, entries: Iterable[CorpusEntry] | None = None) -> Score:
    """Run a scanner over the corpus and return its confusion matrix."""
    result = Score()
    for entry in entries if entries is not None else load_corpus():
        flagged = bool(scanner(entry.line, entry.stream))
        if entry.error_signal and flagged:
            result.true_positive += 1
        elif entry.error_signal:
            result.false_negative += 1
            result.missed.append(entry.id)
        elif flagged:
            result.false_positive += 1
            result.spurious.append(entry.id)
        else:
            result.true_negative += 1
    return result


def deterministic_scanner(line: str, stream: str) -> bool:
    """The shipped pattern set, adapted to the one-line scanner signature."""
    from gates import log_checks

    return bool(log_checks.scan(line, stream))


def model_scanner(model_fn, *, few_shot: int = 3, **layer_kw) -> ScannerFn:
    """The model tier as a line-level scanner, for measurement only.

    One call per line is not how ``llm_scan`` runs in the gate — there it sends
    the unflagged lines in one batch — but scoring needs a per-line verdict, and
    a batch that returns indices cannot be attributed line by line without it.
    The prompt is the shipped one, so what is measured is the shipped judgement,
    at a cost this is not asking anyone to pay in production.
    """
    from gates.llm import ModelLayer
    from gates.llm_scan import scan_with_model

    layer = ModelLayer(model_fn, **layer_kw)

    def scan_one(line: str, stream: str) -> bool:
        text = line if stream == "stdout" else ""
        err = line if stream == "stderr" else ""
        return bool(
            scan_with_model(layer, text, err, few_shot=few_shot).findings
        )

    return scan_one


def combined_scanner(model_fn, *, few_shot: int = 3, **layer_kw) -> ScannerFn:
    """Deterministic first, model only on what it did not flag.

    This is the arrangement the gate actually runs, and therefore the one whose
    precision and recall belong in the paper.
    """
    model = model_scanner(model_fn, few_shot=few_shot, **layer_kw)

    def scan_one(line: str, stream: str) -> bool:
        return deterministic_scanner(line, stream) or model(line, stream)

    return scan_one


def bench_variants(
    model_fn, variants: dict[str, dict] | None = None
) -> str:
    """Score prompt variants against the corpus, side by side.

    This is what makes prompt engineering an engineering activity rather than a
    matter of taste: each change is a row, and precision is the column that
    decides. A variant that lifts recall while precision falls has not improved
    the scanner, it has moved the cost from missed signals to wasted rewrites.

    Needs a real model. Each variant is one pass over the corpus — 68 calls.
    """
    variants = variants or {
        "no few-shot": {"few_shot": 0},
        "few-shot k=2": {"few_shot": 2},
        "few-shot k=3": {"few_shot": 3},
    }
    baseline = score(deterministic_scanner)
    out = [render("deterministic only (baseline)", baseline), ""]
    for name, kwargs in variants.items():
        result = score(combined_scanner(model_fn, **kwargs))
        out.append(render(f"deterministic + model, {name}", result))
        out.append("")
    out.append(
        "  Precision is the floor. A variant that raises recall while precision\n"
        "  falls has moved the cost from missed signals to wasted rewrites, and\n"
        "  the engineer pays either way."
    )
    return "\n".join(out)


def render(name: str, result: Score) -> str:
    lo_p, hi_p = result.interval("precision")
    lo_r, hi_r = result.interval("recall")
    return "\n".join(
        [
            f"{name}",
            f"  precision  {result.precision:.3f}  "
            f"(95% CI {lo_p:.3f}–{hi_p:.3f})   "
            f"{result.true_positive} of {result.flagged} flagged were real",
            f"  recall     {result.recall:.3f}  "
            f"(95% CI {lo_r:.3f}–{hi_r:.3f})   "
            f"{result.true_positive} of {result.positives} signals found",
            f"  f1         {result.f1:.3f}",
            f"  missed     {', '.join(result.missed) or '—'}",
            f"  spurious   {', '.join(result.spurious) or '—'}",
        ]
    )


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    entries = load_corpus()
    print(
        f"corpus: {len(entries)} lines, "
        f"{sum(e.error_signal for e in entries)} signals, "
        f"{sum(e.is_recall_gap for e in entries)} of them outside the pattern set, "
        f"{sum(e.is_hard_negative for e in entries)} hard negatives\n"
    )
    print(render("deterministic (shipped pattern set)", score(deterministic_scanner)))
