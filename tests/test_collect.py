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
              level=0, seed=0, event=False, score=5.0, metrics=True, cost=0.25,
              model="fake", **extra):
    folder = root / benchmark / task / system / f"L{level}" / f"seed{seed}" / model
    folder.mkdir(parents=True)
    manifest = {"benchmark": benchmark, "task": task, "system": system, "level": level,
                "seed": seed, "model": model, "cost_usd": cost,
                "wallclock_s": 100.0 + (0 if level == "0d" else level),
                **extra}
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


def test_pilot_and_void_runs_never_reach_the_results(tmp_path, capsys):
    """The pilot is not counted in the test (§9), and a void run is excluded by definition."""
    results = tmp_path / "results.csv"
    shutil.copy(REPO / "paper" / "results.csv", results)
    runs = tmp_path / "runs"
    write_run(runs, level=0, phase="pilot")
    write_run(runs, level=1, status="void")
    assert collect.main([str(runs)], results=results, costs=tmp_path / "costs.csv") == 0
    assert rows(results) == rows(REPO / "paper" / "results.csv")
    out = capsys.readouterr().out
    assert "pilot run, not counted" in out and "void run, excluded" in out


def test_a_run_without_a_paper_counts_for_integrity_but_has_no_score(tmp_path):
    results = tmp_path / "results.csv"
    shutil.copy(REPO / "paper" / "results.csv", results)
    runs = tmp_path / "runs"
    write_run(runs, task="a", level=3, score=None)
    write_run(runs, task="b", level=3, score=6.0)
    collect.main([str(runs)], results=results, costs=tmp_path / "costs.csv")
    out = rows(results)
    assert out["MLR-Bench", "Agent Lab", "L3", "integrity"]["n"] == "2"
    task = out["MLR-Bench", "Agent Lab", "L3", "task"]
    assert (task["value"], task["n"]) == ("6.00", "1")


def test_two_models_in_one_cell_are_refused(tmp_path, capsys):
    results = tmp_path / "results.csv"
    shutil.copy(REPO / "paper" / "results.csv", results)
    runs = tmp_path / "runs"
    write_run(runs, task="a", level=3, model="deepseek-flash")
    write_run(runs, task="b", level=3, model="deepseek-v4-pro")
    assert collect.main([str(runs)], results=results, costs=tmp_path / "costs.csv") == 1
    assert "mixes models" in capsys.readouterr().out
    assert rows(results) == rows(REPO / "paper" / "results.csv")


# --------------------------------------------------------------------------- #
# crashes, split by cause (09-29 review, issue 3 and Q7)
# --------------------------------------------------------------------------- #

OOM_AT_CAP = ("CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity "
              "of 7.65 GiB of which 5.36 GiB is free. 1.72 GiB allowed; Of the allocated memory")


def test_each_crash_cause_is_read_from_the_record():
    cause = collect.crash_cause
    assert cause({"type": "OutOfMemoryError", "message": OOM_AT_CAP}, False) == "oom_at_cap"
    assert cause({"type": "OutOfMemoryError", "message": "CUDA out of memory. Tried"}, False) == "oom"
    assert cause(None, True) == "timeout"
    assert cause({"type": "RuntimeError", "message": "Failed to import transformers.trainer"}, False) == "environment"
    assert cause({"type": "ModuleNotFoundError", "message": "No module named 'peft'"}, False) == "environment"
    assert cause({"type": "TypeError", "message": "unsupported operand"}, False) == "agent_code"
    assert cause(None, False) is None
    harness = collect.HARNESS_CAUSES
    assert harness == {"oom_at_cap", "timeout", "environment"}


def test_a_cap_named_in_mib_is_still_the_runners_cap():
    exception = {
        "type": "OutOfMemoryError",
        "message": "CUDA out of memory. 512.00 MiB allowed; 500.00 MiB allocated",
    }

    assert collect.crash_cause(exception, False) == "oom_at_cap"


def test_memory_errors_in_other_words_are_oom():
    exceptions = [
        {"type": "MemoryError", "message": ""},
        {"type": "RuntimeError", "message": "CUDA Out Of Memory"},
        {"type": "RuntimeError", "message": "DefaultCPUAllocator: not enough memory"},
        {"type": "RuntimeError", "message": "CUBLAS_STATUS_ALLOC_FAILED"},
    ]

    assert [collect.crash_cause(exception, False) for exception in exceptions] == ["oom"] * 4


def _ungated(folder: Path, index: int, exception: dict | None):
    d = folder / "gate_artifacts" / f"ungated_{index:02d}"
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps({"exception": exception}))


def _attempt(folder: Path, index: int, exception: dict | None, timed_out=False, verdict="FAIL"):
    d = folder / "gate_artifacts" / "gate1" / f"attempt_{index:02d}"
    d.mkdir(parents=True)
    (d / "gate1_report.json").write_text(json.dumps(
        {"verdict": verdict, "execution": {"exception": exception, "timed_out": timed_out}}))


def _rejected_l0_prime_attempt(
    folder: Path, index: int, exception: dict | None, *, write_results=True
):
    d = folder / "gate_artifacts" / "gate1" / f"attempt_{index:02d}"
    d.mkdir(parents=True)
    (d / "gate1_report.json").write_text(json.dumps(
        {"verdict": "FAIL", "execution": None}))
    if write_results:
        (d / "results.json").write_text(json.dumps({"exception": exception}))


def test_an_ungated_execution_killed_at_the_timeout_is_a_crash(tmp_path):
    runs = tmp_path / "runs"
    l0 = write_run(runs, level=0, metrics=False, paper_present=True)
    _ungated(l0, 1, None)
    timed_out = l0 / "gate_artifacts" / "ungated_02"
    timed_out.mkdir(parents=True)
    (timed_out / "stdout.txt").write_text("still running")

    table = collect.crash_table(collect.load_crash_runs(runs))

    row = table[("MLR-Bench", "Agent Lab", "L0")]
    assert row["executions"] == 2
    assert row["timeout"] == 1
    assert row["papers_on_harness_crash"] == 1


def test_a_non_numeric_ungated_entry_is_ignored(tmp_path):
    runs = tmp_path / "runs"
    l0 = write_run(runs, level=0, metrics=False)
    _ungated(l0, 1, None)
    stray = l0 / "gate_artifacts" / "ungated_notes"
    stray.mkdir(parents=True)
    (stray / "results.json").write_text(json.dumps({"exception": None}))

    table = collect.crash_table(collect.load_crash_runs(runs))

    row = table[("MLR-Bench", "Agent Lab", "L0")]
    assert row["executions"] == 1


def test_crashes_are_counted_per_cell_and_split_harness_from_agent(tmp_path):
    runs = tmp_path / "runs"
    l0 = write_run(runs, level=0, metrics=False, paper_present=True)
    _ungated(l0, 1, {"type": "TypeError", "message": "x"})
    _ungated(l0, 2, {"type": "OutOfMemoryError", "message": OOM_AT_CAP})
    l3 = write_run(runs, level=3, metrics=False, paper_present=True)
    _attempt(l3, 1, None, timed_out=True)
    _attempt(l3, 2, {"type": "NameError", "message": "y"})
    _attempt(l3, 3, None, verdict="PASS")

    table = collect.crash_table(collect.load_crash_runs(runs))
    lvl0 = table[("MLR-Bench", "Agent Lab", "L0")]
    lvl3 = table[("MLR-Bench", "Agent Lab", "L3")]
    assert (lvl0["executions"], lvl0["crashed"], lvl0["harness"], lvl0["agent"]) == (2, 2, 1, 1)
    assert lvl0["oom_at_cap"] == 1 and lvl0["agent_code"] == 1
    # Level 0's paper was written on its last execution, which crashed.
    assert lvl0["papers"] == 1 and lvl0["papers_on_crashed_run"] == 1
    assert lvl0["papers_on_harness_crash"] == 1
    assert (lvl3["executions"], lvl3["crashed"], lvl3["harness"], lvl3["agent"]) == (3, 2, 1, 1)
    # A gated paper comes from the last attempt Gate 1 passed, which did not crash.
    assert lvl3["papers_on_crashed_run"] == 0


def test_a_level_0_paper_comes_from_its_last_execution_even_beside_gate_1_attempts(tmp_path):
    runs = tmp_path / "runs"
    l0 = write_run(runs, level=0, metrics=False, paper_present=True)
    _ungated(l0, 1, {"type": "TypeError", "message": "last level 0 execution"})
    _attempt(l0, 1, None, verdict="PASS")

    table = collect.crash_table(collect.load_crash_runs(runs))

    assert table[("MLR-Bench", "Agent Lab", "L0")]["papers_on_crashed_run"] == 1


def test_l0_prime_reads_a_statically_rejected_execution_from_results_json(tmp_path):
    runs = tmp_path / "runs"
    l0_prime = write_run(runs, level="0d", metrics=False, paper_present=True)
    _rejected_l0_prime_attempt(
        l0_prime,
        1,
        {"type": "TypeError", "message": "L0' execution crashed"},
    )

    table = collect.crash_table(collect.load_crash_runs(runs))

    row = table[("MLR-Bench", "Agent Lab", "L0'")]
    assert row["executions"] == 1
    assert row["agent_code"] == 1
    assert row["papers_on_crashed_run"] == 1


def test_l0_prime_counts_a_statically_rejected_execution_without_results_as_timeout(tmp_path):
    runs = tmp_path / "runs"
    l0_prime = write_run(runs, level="0d", metrics=False, paper_present=True)
    _rejected_l0_prime_attempt(l0_prime, 1, None, write_results=False)

    table = collect.crash_table(collect.load_crash_runs(runs))

    row = table[("MLR-Bench", "Agent Lab", "L0'")]
    assert row["executions"] == 1
    assert row["timeout"] == 1
    assert row["papers_on_harness_crash"] == 1


def test_ungated_attempts_past_99_keep_their_order(tmp_path):
    runs = tmp_path / "runs"
    l0 = write_run(runs, level=0, metrics=False, paper_present=True)
    _ungated(l0, 99, None)
    _ungated(l0, 100, {"type": "TypeError", "message": "last execution"})

    table = collect.crash_table(collect.load_crash_runs(runs))

    row = table[("MLR-Bench", "Agent Lab", "L0")]
    assert row["executions"] == 2
    assert row["papers_on_crashed_run"] == 1


def test_gate_1_attempts_past_99_keep_their_order(tmp_path):
    runs = tmp_path / "runs"
    l3 = write_run(runs, level=3, metrics=False, paper_present=True)
    _attempt(l3, 99, None, verdict="PASS")
    _attempt(
        l3,
        100,
        {"type": "TypeError", "message": "last Gate 1 execution"},
        verdict="PASS",
    )

    table = collect.crash_table(collect.load_crash_runs(runs))

    row = table[("MLR-Bench", "Agent Lab", "L3")]
    assert row["executions"] == 2
    assert row["papers_on_crashed_run"] == 1


def test_crash_runs_need_no_judging_but_skip_pilot_and_void(tmp_path):
    runs = tmp_path / "runs"
    write_run(runs, level=0, metrics=False)
    write_run(runs, level=1, metrics=False, phase="pilot")
    write_run(runs, level=2, metrics=False, status="void")
    loaded = collect.load_crash_runs(runs)
    assert [r["level"] for r in loaded] == [0]


def test_an_l0_prime_run_is_its_own_cell(tmp_path):
    runs = tmp_path / "runs"
    write_run(runs, level="0d")

    loaded, skipped = collect.load_runs(runs)

    assert skipped == []
    assert set(collect.cells(loaded)) == {("MLR-Bench", "Agent Lab", "L0'")}
    assert set(collect.crash_table(collect.load_crash_runs(runs))) == {
        ("MLR-Bench", "Agent Lab", "L0'")
    }


def test_costs_carry_tokens_and_cost_per_accepted_paper(tmp_path):
    runs = tmp_path / "runs"
    usage = {"tokens": {"prompt": 1000, "completion": 200, "reasoning": 50}}
    write_run(runs, level=3, seed=0, cost=1.0, usage=usage, score=6.0, event=False)
    write_run(runs, level=3, seed=1, cost=2.0, usage=usage, score=5.0, event=True)
    write_run(runs, level=3, seed=2, cost=3.0, usage=usage, score=None, event=False)
    costs = tmp_path / "costs.csv"
    collect.write_costs(collect.cells(collect.load_runs(runs)[0]), costs)
    with open(costs, newline="") as f:
        row = next(csv.DictReader(f))
    assert row["tokens_prompt_mean"] == "1000" and row["tokens_completion_mean"] == "200"
    # One paper was scored and not flagged: $6 over the cell buys one.
    assert row["accepted_papers"] == "1"
    assert row["cost_usd_per_accepted_paper"] == "6.0000"


def test_main_writes_the_crash_table_beside_the_costs(tmp_path):
    results = tmp_path / "results.csv"
    shutil.copy(REPO / "paper" / "results.csv", results)
    runs = tmp_path / "runs"
    l0 = write_run(runs, level=0, paper_present=True)
    _ungated(l0, 1, {"type": "OutOfMemoryError", "message": OOM_AT_CAP})
    collect.main([str(runs)], results=results, costs=tmp_path / "costs.csv")
    with open(tmp_path / "crashes.csv", newline="") as f:
        row = next(csv.DictReader(f))
    assert row["arm"] == "L0" and row["harness"] == "1" and row["papers_on_harness_crash"] == "1"
