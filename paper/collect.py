"""Turn finished runs into measured rows of results.csv, and their cost into costs.csv.

    python3 paper/collect.py RUNS_ROOT

Reads the layout in ``paper/PLAN.md`` §6. Each run folder must hold two files:

``manifest.json``, written by the level runner::

    {"benchmark": "MLR-Bench", "task": "...", "system": "Agent Lab",
     "level": 3, "seed": 0, "model": "deepseek-flash",
     "cost_usd": 0.41, "wallclock_s": 812.5, ...}

``metrics.json``, written once the judge has labelled the run::

    {"integrity_event": false, "task_score": 6.5}

``integrity_event`` is the benchmark's integrity outcome from §3: a faked
result, an untraceable answer, or a manipulated manuscript accepted.
``task_score`` is the MLR-Judge overall score on MLR-Bench, and 0 or 1
elsewhere. A run folder missing either file has not finished and is skipped
with a line saying so, never counted as a pass.

Each (benchmark, system, level) cell replaces its two rows in results.csv with
``status`` = ``measured``. A rate is a percentage with its Wilson interval; the
MLR-Judge score is a mean with a seeded paired-bootstrap interval. A cell with
no row in results.csv is reported and not written: the plan and the runs
disagree, and that is for a person to settle.

Cost is measured, not capped: costs.csv holds each cell's runs, total and mean
dollars, and mean wallclock, which is M6's input.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from rig.stats import paired_bootstrap_ci, wilson  # noqa: E402

RESULTS = HERE / "results.csv"
COSTS = HERE / "costs.csv"

#: How each benchmark's task score aggregates: a mean on a scale, or a rate.
MEAN_SCORED = {"MLR-Bench"}

REQUIRED = ("benchmark", "task", "system", "level", "seed")


def load_runs(root: Path) -> tuple[list[dict], list[str]]:
    """Every finished run under ``root``, and a note for each unfinished one."""
    runs: list[dict] = []
    skipped: list[str] = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        folder = manifest_path.parent
        metrics_path = folder / "metrics.json"
        if not metrics_path.exists():
            skipped.append(f"{folder.relative_to(root)}: no metrics.json yet")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        missing = [k for k in REQUIRED if k not in manifest]
        if missing:
            skipped.append(f"{folder.relative_to(root)}: manifest lacks {', '.join(missing)}")
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        runs.append({**manifest, **metrics, "folder": str(folder)})
    return runs, skipped


def cells(runs: list[dict]) -> dict[tuple[str, str, str], list[dict]]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for run in runs:
        grouped[(run["benchmark"], run["system"], f"L{int(run['level'])}")].append(run)
    return dict(grouped)


def _rate_row(events: int, n: int) -> dict[str, str]:
    lo, hi = wilson(events, n)
    return {
        "value": f"{100 * events / n:.1f}",
        "ci_low": f"{100 * lo:.1f}",
        "ci_high": f"{100 * hi:.1f}",
        "n": str(n),
    }


def measure(benchmark: str, runs: list[dict]) -> dict[str, dict[str, str]]:
    """The cell's integrity and task rows, as the CSV spells them."""
    n = len(runs)
    integrity = _rate_row(sum(bool(r["integrity_event"]) for r in runs), n)
    scores = [float(r["task_score"]) for r in runs]
    if benchmark in MEAN_SCORED:
        mean, lo, hi = paired_bootstrap_ci(scores)
        task = {"value": f"{mean:.2f}", "ci_low": f"{lo:.2f}", "ci_high": f"{hi:.2f}", "n": str(n)}
    else:
        task = _rate_row(sum(1 for s in scores if s >= 1), n)
    return {"integrity": integrity, "task": task}


def update_results(
    grouped: dict[tuple[str, str, str], list[dict]], path: Path = RESULTS
) -> tuple[int, list[str]]:
    """Rewrite the measured rows in place. Returns rows written and unmatched cells."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    index = {(r["benchmark"], r["system"], r["arm"], r["metric"]): r for r in rows}

    written = 0
    unmatched: list[str] = []
    for (benchmark, system, arm), runs in sorted(grouped.items()):
        measured = measure(benchmark, runs)
        if not all((benchmark, system, arm, m) in index for m in measured):
            unmatched.append(f"{benchmark} / {system} / {arm} ({len(runs)} runs)")
            continue
        for metric, values in measured.items():
            row = index[(benchmark, system, arm, metric)]
            row.update(values)
            row["status"] = "measured"
            row["source"] = f"paper/collect.py, {len(runs)} runs"
            written += 1

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return written, unmatched


def write_costs(grouped: dict[tuple[str, str, str], list[dict]], path: Path = COSTS) -> None:
    """Measured spend and wallclock per cell. A run that did not record one is counted as missing."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([
            "benchmark", "system", "arm", "runs", "cost_usd_total", "cost_usd_mean",
            "wallclock_s_mean", "runs_missing_cost",
        ])
        for (benchmark, system, arm), runs in sorted(grouped.items()):
            costs = [float(r["cost_usd"]) for r in runs if r.get("cost_usd") is not None]
            clocks = [float(r["wallclock_s"]) for r in runs if r.get("wallclock_s") is not None]
            writer.writerow([
                benchmark, system, arm, len(runs),
                f"{sum(costs):.4f}" if costs else "",
                f"{sum(costs) / len(costs):.4f}" if costs else "",
                f"{sum(clocks) / len(clocks):.1f}" if clocks else "",
                len(runs) - len(costs),
            ])


def main(argv: list[str] | None = None, *, results: Path = RESULTS, costs: Path = COSTS) -> int:
    parser = argparse.ArgumentParser(prog="python3 paper/collect.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("runs_root", type=Path)
    args = parser.parse_args(argv)

    runs, skipped = load_runs(args.runs_root)
    grouped = cells(runs)
    written, unmatched = update_results(grouped, results)
    write_costs(grouped, costs)

    print(f"{len(runs)} finished runs in {len(grouped)} cells; {written} rows of {results.name} now measured")
    for note in skipped:
        print(f"  skipped {note}")
    for cell in unmatched:
        print(f"  no row in {results.name} for {cell}: not written")
    return 1 if unmatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
