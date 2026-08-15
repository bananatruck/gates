"""What each arm's report actually says, and whether it survives re-execution.

The gated arm's report is its registry: a typed value per key, bound to a run.
The ungated arm has no registry — but it does have a report, and pretending
otherwise made the comparison unanswerable. Its report is whatever numbers
appear in the 1,000 characters upstream hands the writing agent, because those
are the only numbers the writer can copy.

So both arms are measured the same way, on the same three questions:

1. **Does the report contain the results at all?** For the gated arm the
   registry is complete by construction. For the ungated arm the results may
   fall outside the 1,000-character window entirely — measured, not assumed.
2. **Can each number be tied to an execution?** A registry value carries a trace
   id and a code hash. A printed number carries nothing.
3. **Does it reproduce?** Re-run the accepted code and compare.

Point (3) is the one that makes this a comparison rather than an assertion, and
it is worth being careful about what a pass there means. A printed number that
reproduces is *correct and untraceable*: nothing binds it to the run that
produced it, so a later edit, a different seed, or a number copied from an
earlier attempt is indistinguishable from a measurement. Reproducibility and
provenance are different properties, and the ungated arm can only ever have the
first by luck.

Runs entirely on artifacts already on disk. No model calls.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from gates import run_experiment

LEGACY_MAX_LEN = 1000

#: "Test accuracy with all training data: 0.9400", "Ratio (full/subset): 1.0000",
#: "final_acc = 0.94". A label, a separator, a number — which is the shape a
#: writing agent can actually lift a result from.
_LABELLED_NUMBER = re.compile(
    r"([A-Za-z][A-Za-z0-9 _()/%\-\.]{2,60}?)\s*[:=]\s*(-?\d+\.?\d*)\s*$"
)

#: Lines that are progress, not results. A training curve is not a finding, and
#: counting each epoch as a reported number would drown the comparison.
_PROGRESS = re.compile(r"^\s*(step|epoch|iter|iteration|batch)\b", re.IGNORECASE)


@dataclass
class ArmReport:
    arm: str
    #: Numbers a writer could lift from what it was given.
    visible: dict[str, float] = field(default_factory=dict)
    #: Result-shaped numbers present in the full capture.
    produced: dict[str, float] = field(default_factory=dict)
    reproduced: dict[str, float] = field(default_factory=dict)
    matched: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)
    traceable: int = 0
    note: str = ""

    @property
    def lost_to_channel(self) -> int:
        """Results the run produced that never reached the writer."""
        return max(len(self.produced) - len(self.visible), 0)

    @property
    def reproduction_rate(self) -> float | None:
        total = len(self.matched) + len(self.mismatched)
        return len(self.matched) / total if total else None

    def to_dict(self) -> dict:
        return {
            "arm": self.arm,
            "produced": self.produced,
            "visible": self.visible,
            "reproduced": self.reproduced,
            "matched": self.matched,
            "mismatched": self.mismatched,
            "traceable": self.traceable,
            "lost_to_channel": self.lost_to_channel,
            "reproduction_rate": self.reproduction_rate,
            "note": self.note,
        }


def extract_results(text: str) -> dict[str, float]:
    """Result-shaped numbers in a capture, keyed by their printed label."""
    found: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or _PROGRESS.match(line):
            continue
        m = _LABELLED_NUMBER.match(line)
        if not m:
            continue
        label = " ".join(m.group(1).split()).rstrip(":= ")
        try:
            found[label] = float(m.group(2))
        except ValueError:
            continue
    return found


def _compare(reported: dict[str, float], again: dict[str, float]) -> tuple[list, list]:
    matched, mismatched = [], []
    for key, value in reported.items():
        other = again.get(key)
        if other is None:
            mismatched.append(key)
        elif _timing(key):
            ok = abs(other - value) <= max(0.5 * abs(value), 0.05)
            (matched if ok else mismatched).append(key)
        else:
            (matched if other == value else mismatched).append(key)
    return matched, mismatched


def _timing(key: str) -> bool:
    k = key.lower()
    return any(t in k for t in ("time", "wallclock", "sec", "_s", "elapsed"))


def analyse_ungated(run_dir: Path, accepted_turn: int, *, timeout_s: int = 240) -> ArmReport:
    """The ungated arm's report: what the writer sees, and whether it holds."""
    attempt = run_dir / "ungated" / f"attempt_{accepted_turn:02d}"
    report = ArmReport(arm="without Gate 1")
    stdout_path = attempt / "stdout.txt"
    if not stdout_path.exists():
        report.note = "no capture on disk"
        return report

    full = stdout_path.read_text(errors="replace")
    report.produced = extract_results(full)
    report.visible = extract_results(full[:LEGACY_MAX_LEN])
    # Nothing binds a printed number to the run that printed it.
    report.traceable = 0

    code = (attempt / "experiment.py").read_text(errors="replace")
    execution = run_experiment(code, run_dir / "ungated" / "verify_report",
                               timeout_s=timeout_s)
    if execution.exit_code != 0:
        report.note = f"re-execution exited {execution.exit_code}"
        return report
    report.reproduced = extract_results(execution.stdout_text())
    report.matched, report.mismatched = _compare(report.visible, report.reproduced)
    return report


def analyse_gated(run_dir: Path, accepted_turn: int) -> ArmReport:
    """The gated arm's report is its registry, already verified by the gate."""
    report = ArmReport(arm="with Gate 1")
    reg_path = (run_dir / "gated" / "gate1" / f"attempt_{accepted_turn:02d}"
                / "registry.json")
    if not reg_path.exists():
        report.note = "no registry on disk"
        return report
    registry = json.loads(reg_path.read_text())
    values = registry.get("values") or registry.get("metrics") or {}
    report.produced = {k: v.get("value") for k, v in values.items()
                       if isinstance(v, dict)}
    # The registry is what the writer receives, in full: nothing is truncated
    # away, which is the entire point of replacing the 1,000-character channel.
    report.visible = dict(report.produced)
    report.traceable = sum(
        1 for v in values.values()
        if isinstance(v, dict) and v.get("trace_id")
    )
    return report


def analyse_run(run_dir: str | Path, *, timeout_s: int = 240) -> dict:
    run_dir = Path(run_dir)
    data = json.loads((run_dir / "ablation.json").read_text())
    out = {"run": run_dir.name}

    g = data["arms"]["gated"]
    if g["accepted"]:
        gr = analyse_gated(run_dir, g["accepted_at"])
        v = g.get("verification") or {}
        gr.reproduced = v.get("reproduced", {})
        gr.matched = v.get("matched", [])
        gr.mismatched = v.get("mismatched", [])
        out["gated"] = gr.to_dict()

    u = data["arms"]["ungated"]
    if u["accepted"]:
        out["ungated"] = analyse_ungated(
            run_dir, u["accepted_at"], timeout_s=timeout_s
        ).to_dict()
    return out


def analyse_all(root: str | Path, *, timeout_s: int = 240) -> dict[str, dict]:
    """Every run under ``root`` that has an ablation record."""
    root = Path(root)
    out: dict[str, dict] = {}
    for path in sorted(root.glob("*/ablation.json")):
        out[path.parent.name] = analyse_run(path.parent, timeout_s=timeout_s)
    return out


if __name__ == "__main__":
    import sys

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/home/kesh/AgentLaboratory-Gemini/ablation_runs"
    )
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "report_accuracy.json"
    results = analyse_all(root)
    dest.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name, r in results.items():
        g, u = r.get("gated", {}), r.get("ungated", {})
        print(f"{name}: gated {len(g.get('visible', {}))} visible / "
              f"{g.get('traceable', 0)} traceable | "
              f"ungated {len(u.get('produced', {}))} produced / "
              f"{len(u.get('visible', {}))} visible / "
              f"{u.get('lost_to_channel', 0)} lost / "
              f"{len(u.get('matched', []))} reproduced")
    print(f"\nwrote {dest}")
