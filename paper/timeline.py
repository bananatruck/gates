"""Turn each pilot run's folder into a timeline: spend over time and every gate attempt.

    python3 paper/timeline.py ~/gates-runs/runs [OUT]   # default OUT .cache/paper/timeline.csv

One row per event. A model call carries the run's cumulative cost and tokens at
that moment; cost is recomputed from the call's own token counts and the prices
the run's manifest recorded, and the run is refused unless the recomputed total
matches the manifest's measured cost. A gate attempt or an ungated execution
carries its outcome: Gate 1 attempts their execution's finish time, and Gate 2
and 3 attempts and ungated executions the time their report or results file was
written. A run ends wallclock_s after its start (the manifest's start_utc and
end_utc are the wave's), and an event outside the run is refused.

The output is wave data, so it is written under .cache/ and never committed;
only the figure drawn from it is. Stdlib only.
"""

import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent / ".cache" / "paper" / "timeline.csv"
FIELDS = ["level", "seed", "t_h", "cost_usd", "tokens_m", "event", "outcome"]
COST_TOLERANCE_USD = 1e-4
# A report file can be written a moment after the run clock stops.
END_SLACK_S = 60


def _utc(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(timezone.utc)


def _mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def call_cost(entry: dict, prices: dict) -> float:
    """One call's cost: cache hits, cache misses and output at the served model's price.

    Calls outside the manifest's weekday peak hours (UTC) pay the off-peak factor.
    """
    usage = entry["usage"]
    table = prices["peak_per_million"][entry.get("served_model") or entry["requested_model"]]
    hit = usage.get("prompt_cache_hit_tokens") or 0
    miss = usage.get("prompt_cache_miss_tokens")
    if miss is None:
        miss = usage["prompt_tokens"] - hit
    cost = (hit * table["cache_hit"] + miss * table["cache_miss"]
            + usage["completion_tokens"] * table["output"]) / 1e6
    when = _utc(entry["ts_utc"])
    peak = when.weekday() < 5 and any(
        start <= when.hour < end for start, end in prices["peak_hours_utc_weekdays"]
    )
    return cost if peak else cost * prices["off_peak_factor"]


def run_events(folder: Path) -> list[dict]:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    # start_utc and end_utc are the wave's; a run ends wallclock_s after the start.
    start = _utc(manifest["start_utc"])
    end = start + timedelta(seconds=float(manifest["wallclock_s"]))
    level = f"L{manifest['level']}"
    seed = str(manifest["seed"])

    def hours(moment: datetime) -> float:
        if not start <= moment <= end + timedelta(seconds=END_SLACK_S):
            raise ValueError(f"{folder}: event at {moment.isoformat()} outside the run")
        return round((moment - start).total_seconds() / 3600, 4)

    rows: list[dict] = []
    cost = tokens = 0.0
    for line in (folder / "usage.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        cost += call_cost(entry, manifest["prices"])
        tokens += entry["usage"]["prompt_tokens"] + entry["usage"]["completion_tokens"]
        rows.append({"t": _utc(entry["ts_utc"]), "event": "call", "outcome": ""})
        rows[-1].update(cost_usd=round(cost, 6), tokens_m=round(tokens / 1e6, 4))
    if abs(cost - float(manifest["cost_usd"])) > COST_TOLERANCE_USD:
        raise ValueError(
            f"{folder}: recomputed cost {cost:.6f} != manifest {manifest['cost_usd']}"
        )

    artifacts = folder / "gate_artifacts"
    for results in sorted(artifacts.glob("ungated_*/results.json")):
        record = json.loads(results.read_text(encoding="utf-8"))
        rows.append({"t": _mtime(results), "event": "exec",
                     "outcome": "crash" if record.get("exception") else "ran"})
    for report in sorted(artifacts.glob("gate1/attempt_*/gate1_report.json")):
        data = json.loads(report.read_text(encoding="utf-8"))
        finished = (data.get("execution") or {}).get("finished_at")
        moment = _utc(finished) if finished else _mtime(report)
        rows.append({"t": moment, "event": "gate1", "outcome": data["verdict"].lower()})
    for gate in ("gate2", "gate3"):
        for report in sorted(artifacts.glob(f"{gate}/attempt_*/{gate}_report.json")):
            data = json.loads(report.read_text(encoding="utf-8"))
            rows.append({"t": _mtime(report), "event": gate, "outcome": data["verdict"].lower()})

    rows.sort(key=lambda r: r["t"])
    last_cost = last_tokens = 0.0
    out = []
    for row in rows:
        if row["event"] == "call":
            last_cost, last_tokens = row["cost_usd"], row["tokens_m"]
        out.append({"level": level, "seed": seed, "t_h": hours(row["t"]),
                    "cost_usd": last_cost, "tokens_m": last_tokens,
                    "event": row["event"], "outcome": row["outcome"]})
    out.append({"level": level, "seed": seed, "t_h": hours(end), "cost_usd": last_cost,
                "tokens_m": last_tokens, "event": "end", "outcome": manifest.get("status", "")})
    return out


def pilot_runs(root: Path) -> list[Path]:
    """The eight runs the paper's tables count: phase main, not void."""
    folders = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("phase") != "main" or manifest.get("status") == "void":
            continue
        folders.append(manifest_path.parent)
    return folders


def write(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    root = Path(sys.argv[1]).expanduser()
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    folders = pilot_runs(root / "MLR-Bench")
    events = [row for folder in folders for row in run_events(folder)]
    write(events, target)
    print(f"wrote {len(events)} events from {len(folders)} runs to {target}")
