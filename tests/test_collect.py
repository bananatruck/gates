"""paper/collect.py on a fake run tree: phase 1's done-criterion, held by a test.

"Done when one fake task runs at all four levels and collect.py turns its rows
from dummy to measured" (``paper/PLAN.md`` §5).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("collect", REPO / "paper" / "collect.py")
collect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(collect)


def write_run(root: Path, *, benchmark="MLR-Bench", task="t1", system="Agent Lab",
              level=0, seed=0, event=False, score=5.0, metrics=True, cost=0.25):
    folder = root / benchmark / task / system / f"L{level}" / f"seed{seed}"
    folder.mkdir(parents=True)
    manifest = {"benchmark": benchmark, "task": task, "system": system, "level": level,
                "seed": seed, "model": "fake", "cost_usd": cost, "wallclock_s": 100.0 + level}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    if metrics:
        (folder / "metrics.json").write_text(
            json.dumps({"integrity_event": event, "task_score": score}))
    return folder


def rows(path: Path) -> dict[tuple, dict]:
    with open(path, newline="") as f:
        return {(r["benchmark"], r["system"], r["arm"], r["metric"]): r for r in csv.DictReader(f)}


def test_one_task_at_four_levels_turns_eight_rows_measured(tmp_path, capsys):
    results = tmp_path / "results.csv"
    shutil.copy(REPO / "paper" / "results.csv", results)
    runs = tmp_path / "runs"
    for level in range(4):
        write_run(runs, level=level, event=level == 0, score=5.0 + level / 10)

    code = collect.main([str(runs)], results=results, costs=tmp_path / "costs.csv")
    assert code == 0
    out = rows(results)
    measured = {k for k, r in out.items() if r["status"] == "measured"}
    assert measured == {("MLR-Bench", "Agent Lab", f"L{lv}", m)
                        for lv in range(4) for m in ("integrity", "task")}
    assert out["MLR-Bench", "Agent Lab", "L0", "integrity"]["value"] == "100.0"
    assert out["MLR-Bench", "Agent Lab", "L3", "integrity"]["value"] == "0.0"
    assert out["MLR-Bench", "Agent Lab", "L3", "task"]["value"] == "5.30"
    # Every other row is untouched.
    before = rows(REPO / "paper" / "results.csv")
    assert all(out[k] == before[k] for k in before if k not in measured)
    assert "8 rows of results.csv now measured" in capsys.readouterr().out


def test_a_rate_benchmark_reports_a_percentage_with_its_wilson_interval(tmp_path):
    results = tmp_path / "results.csv"
    shutil.copy(REPO / "paper" / "results.csv", results)
    runs = tmp_path / "runs"
    for task in ("a", "b", "c", "d"):
        write_run(runs, benchmark="CORE-Bench", task=task, level=1,
                  event=task == "a", score=1 if task in "ab" else 0)
    collect.main([str(runs)], results=results, costs=tmp_path / "costs.csv")
    row = rows(results)["CORE-Bench", "Agent Lab", "L1", "task"]
    assert (row["value"], row["n"]) == ("50.0", "4")
    assert float(row["ci_low"]) < 50.0 < float(row["ci_high"])


def test_an_unfinished_run_is_skipped_and_said(tmp_path, capsys):
    results = tmp_path / "results.csv"
    shutil.copy(REPO / "paper" / "results.csv", results)
    runs = tmp_path / "runs"
    write_run(runs, level=2, metrics=False)
    collect.main([str(runs)], results=results, costs=tmp_path / "costs.csv")
    assert rows(results)["MLR-Bench", "Agent Lab", "L2", "task"]["status"] == "dummy"
    assert "no metrics.json yet" in capsys.readouterr().out


def test_a_cell_the_plan_does_not_have_is_reported_not_written(tmp_path, capsys):
    results = tmp_path / "results.csv"
    shutil.copy(REPO / "paper" / "results.csv", results)
    runs = tmp_path / "runs"
    write_run(runs, system="Some Other System", level=1)
    assert collect.main([str(runs)], results=results, costs=tmp_path / "costs.csv") == 1
    assert "no row in results.csv for MLR-Bench / Some Other System / L1" in capsys.readouterr().out
    assert rows(results) == rows(REPO / "paper" / "results.csv")


def test_cost_is_measured_per_cell(tmp_path):
    runs = tmp_path / "runs"
    write_run(runs, task="a", level=3, cost=0.5)
    write_run(runs, task="b", level=3, cost=None)
    results = tmp_path / "results.csv"
    shutil.copy(REPO / "paper" / "results.csv", results)
    collect.main([str(runs)], results=results, costs=tmp_path / "costs.csv")
    with open(tmp_path / "costs.csv", newline="") as f:
        (row,) = list(csv.DictReader(f))
    assert (row["runs"], row["cost_usd_total"], row["runs_missing_cost"]) == ("2", "0.5000", "1")
    assert row["wallclock_s_mean"] == "103.0"
