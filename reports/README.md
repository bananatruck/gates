# Gate 1 report and deck

Two artifacts, one set of numbers:

| file | what it is |
|---|---|
| `GATE1_REPORT.pdf` | the extensive report — checks, metrics, logs, runs, impact, benchmark comparison, defects, limits |
| `GATE1_DECK.pptx` | the same material as a deck, one claim per slide |
| `assets/*.png` | figures, generated from measurements |

## Rebuilding

```bash
python reports/make_charts.py     # figures from recorded measurements
python reports/make_report.py     # HTML -> PDF (via libreoffice)
python reports/make_deck.py       # PPTX
```

## Where the numbers come from

Nothing in either artifact is illustrative. Each figure is either a measurement
recorded in this repository or a number quoted from a paper in the source set,
and each is attributed at the point of use.

Measurements in this repository:

- check inventory — counted from `gates/gate1.py`
- scanner precision/recall — `rig/corpus.py` against `tests/fixtures/log_corpus.jsonl`
- prompt compression — `gates/log_digest.py`, measured on a 203-line capture
- ablation — `rig/ablation.py`, one live run against a local `qwen3:8b`
- call and token split — instrumented over five executions of one solver phase

Published sources (in `Documents/AI Research/Sources`):

- MLR-Bench, arXiv 2505.19955 — hallucination taxonomy and frequencies
- BadScientist, arXiv 2510.18003 — fabricated-paper acceptance rate
- CORE-Bench, arXiv 2409.11363 — reproducibility accuracy, prediction-interval tolerance
- PaperBench, arXiv 2504.01848 — replication scores
- RE-Bench, arXiv 2411.15114 — frontier R&D against human experts

## Colour

The categorical palette is validated rather than chosen by eye: worst adjacent
pair deutan ΔE 9.2, normal-vision ΔE 27.6 on the light surface. The aqua slot
falls below 3:1 contrast against that surface, so every bar using it carries a
visible value label.
