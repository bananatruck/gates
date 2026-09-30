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
elsewhere; it is null for a run that emitted no paper, which then counts in
the integrity denominator and not in the score. A run folder missing either
file has not finished and is skipped with a line saying so, never counted as a
pass.

Two kinds of run never reach the results, and each is named as it is skipped:
a pilot run (``"phase": "pilot"``), because §9 does not count the pilot in the
test, and a void run (``"status": "void"``), which faced conditions its pair
did not. A cell whose runs used more than one model is refused rather than
averaged.

Each (benchmark, system, level) cell replaces its two rows in results.csv with
``status`` = ``measured``. A rate is a percentage with its Wilson interval; the
MLR-Judge score is a mean with a seeded paired-bootstrap interval. A cell with
no row in results.csv is reported and not written: the plan and the runs
disagree, and that is for a person to settle.

Cost is measured, not capped: costs.csv holds each cell's runs, total and mean
dollars, mean wallclock, mean tokens, and the dollars per accepted paper (one
the judges scored and did not flag), which is M6's input.

Crashes are deterministic and need no judging, so crashes.csv, written beside
costs.csv, counts every non-pilot, non-void run: each execution that raised or
timed out, by cause, with the harness's causes apart from the agent's, and how
many papers were written on a run that crashed (09-29 review, issue 3 and Q7).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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

#: A crashed execution's cause, from its record alone.
#: ``oom_at_cap``: out of memory under the runner's per-run GPU share, which
#: PyTorch reports as "N GiB allowed"; ``oom``: out of memory with no cap named;
#: ``timeout``: killed at the execution limit; ``environment``: a package the
#: environment should provide failed to import; ``agent_code``: anything else.
CRASH_CAUSES = ("oom_at_cap", "oom", "timeout", "environment", "agent_code")

#: Causes that belong to the harness rather than to the agent's code. Plain
#: ``oom`` is not among them: without a cap in the message, the model the
#: agent chose may simply not fit the card.
HARNESS_CAUSES = {"oom_at_cap", "timeout", "environment"}

_ALLOWED = re.compile(r"GiB allowed")
_ENVIRONMENT = re.compile(r"Failed to import|No module named|cannot import name")


def crash_cause(exception: dict | None, timed_out: bool) -> str | None:
    """Why an execution crashed, or ``None`` when it did not."""
    if timed_out:
        return "timeout"
    if not exception:
        return None
    kind = str(exception.get("type", ""))
    message = str(exception.get("message", ""))
    if kind == "OutOfMemoryError" or "out of memory" in message:
        return "oom_at_cap" if _ALLOWED.search(message) else "oom"
    if kind in {"ModuleNotFoundError", "ImportError"} or _ENVIRONMENT.search(message):
        return "environment"
    return "agent_code"


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
        if manifest.get("phase") == "pilot":
            skipped.append(f"{folder.relative_to(root)}: pilot run, not counted (§9)")
            continue
        if manifest.get("status") == "void":
            skipped.append(f"{folder.relative_to(root)}: void run, excluded")
            continue
        missing = [k for k in REQUIRED if k not in manifest]
        if missing:
            skipped.append(f"{folder.relative_to(root)}: manifest lacks {', '.join(missing)}")
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        runs.append({**manifest, **metrics, "folder": str(folder)})
    return runs, skipped


def load_crash_runs(root: Path) -> list[dict]:
    """Every run the crash table counts: finished or not judged, never pilot or void."""
    runs: list[dict] = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("phase") == "pilot" or manifest.get("status") == "void":
            continue
        if any(k not in manifest for k in REQUIRED):
            continue
        runs.append({**manifest, "folder": str(manifest_path.parent)})
    return runs


def _executions(folder: Path) -> list[dict]:
    """Each execution a run made, in order: its cause, and whether Gate 1 passed it."""
    artifacts = folder / "gate_artifacts"
    out: list[dict] = []
    for results in sorted(artifacts.glob("ungated_*/results.json")):
        record = json.loads(results.read_text(encoding="utf-8"))
        out.append({"cause": crash_cause(record.get("exception"), False), "passed": None})
    for report_path in sorted((artifacts / "gate1").glob("attempt_*/gate1_report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        execution = report.get("execution") or {}
        out.append({
            "cause": crash_cause(execution.get("exception"), bool(execution.get("timed_out"))),
            "passed": report.get("verdict") == "PASS",
        })
    return out


def _paper_run(executions: list[dict]) -> dict | None:
    """The execution a paper was written from: the last Gate 1 passed, else the last."""
    passed = [e for e in executions if e["passed"]]
    if passed:
        return passed[-1]
    return executions[-1] if executions else None


def crash_table(runs: list[dict]) -> dict[tuple[str, str, str], dict[str, int]]:
    """Per cell: executions, crashes by cause, harness against agent, papers on a crash."""
    table: dict[tuple[str, str, str], dict[str, int]] = {}
    for run in runs:
        key = (run["benchmark"], run["system"], f"L{int(run['level'])}")
        row = table.setdefault(key, {
            "runs": 0, "executions": 0, "crashed": 0, "harness": 0, "agent": 0,
            **dict.fromkeys(CRASH_CAUSES, 0),
            "papers": 0, "papers_on_crashed_run": 0, "papers_on_harness_crash": 0,
        })
        executions = _executions(Path(run["folder"]))
        row["runs"] += 1
        row["executions"] += len(executions)
        for execution in executions:
            cause = execution["cause"]
            if cause is None:
                continue
            row["crashed"] += 1
            row[cause] += 1
            row["harness" if cause in HARNESS_CAUSES else "agent"] += 1
        if run.get("paper_present"):
            row["papers"] += 1
            source = _paper_run(executions)
            if source is not None and source["cause"] is not None:
                row["papers_on_crashed_run"] += 1
                if source["cause"] in HARNESS_CAUSES:
                    row["papers_on_harness_crash"] += 1
    return table


def write_crashes(table: dict[tuple[str, str, str], dict[str, int]], path: Path) -> None:
    columns = ["runs", "executions", "crashed", "harness", "agent", *CRASH_CAUSES,
               "papers", "papers_on_crashed_run", "papers_on_harness_crash"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["benchmark", "system", "arm", *columns])
        for key in sorted(table):
            writer.writerow([*key, *(table[key][c] for c in columns)])


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
    scores = [float(r["task_score"]) for r in runs if r.get("task_score") is not None]
    if not scores:
        task = {"value": "", "ci_low": "", "ci_high": "", "n": "0"}
    elif benchmark in MEAN_SCORED:
        mean, lo, hi = paired_bootstrap_ci(scores)
        task = {"value": f"{mean:.2f}", "ci_low": f"{lo:.2f}", "ci_high": f"{hi:.2f}",
                "n": str(len(scores))}
    else:
        task = _rate_row(sum(1 for s in scores if s >= 1), len(scores))
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
        models = sorted({str(r.get("model")) for r in runs})
        if len(models) > 1:
            unmatched.append(f"{benchmark} / {system} / {arm} mixes models {models}")
            continue
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
            "wallclock_s_mean", "runs_missing_cost", "tokens_prompt_mean",
            "tokens_completion_mean", "tokens_reasoning_mean", "accepted_papers",
            "cost_usd_per_accepted_paper",
        ])
        for (benchmark, system, arm), runs in sorted(grouped.items()):
            costs = [float(r["cost_usd"]) for r in runs if r.get("cost_usd") is not None]
            clocks = [float(r["wallclock_s"]) for r in runs if r.get("wallclock_s") is not None]
            # Accepted: the judges scored the paper and did not flag it (D63).
            accepted = sum(
                1 for r in runs if r.get("task_score") is not None and not r.get("integrity_event")
            )
            writer.writerow([
                benchmark, system, arm, len(runs),
                f"{sum(costs):.4f}" if costs else "",
                f"{sum(costs) / len(costs):.4f}" if costs else "",
                f"{sum(clocks) / len(clocks):.1f}" if clocks else "",
                len(runs) - len(costs),
                *(_mean_tokens(runs, kind) for kind in ("prompt", "completion", "reasoning")),
                accepted,
                f"{sum(costs) / accepted:.4f}" if costs and accepted else "",
            ])


def _mean_tokens(runs: list[dict], kind: str) -> str:
    counts = [
        ((r.get("usage") or {}).get("tokens") or {}).get(kind) for r in runs
    ]
    counts = [c for c in counts if c is not None]
    return f"{sum(counts) / len(counts):.0f}" if counts else ""


def main(argv: list[str] | None = None, *, results: Path = RESULTS, costs: Path = COSTS) -> int:
    parser = argparse.ArgumentParser(prog="python3 paper/collect.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("runs_root", type=Path)
    args = parser.parse_args(argv)

    runs, skipped = load_runs(args.runs_root)
    grouped = cells(runs)
    written, unmatched = update_results(grouped, results)
    write_costs(grouped, costs)
    write_crashes(crash_table(load_crash_runs(args.runs_root)), costs.with_name("crashes.csv"))

    print(f"{len(runs)} finished runs in {len(grouped)} cells; {written} rows of {results.name} now measured")
    for note in skipped:
        print(f"  skipped {note}")
    for cell in unmatched:
        if "mixes models" in cell:
            print(f"  {cell}: not written")
        else:
            print(f"  no row in {results.name} for {cell}: not written")
    return 1 if unmatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
