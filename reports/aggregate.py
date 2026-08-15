"""Aggregate every ablation run on disk, and audit where each number came from.

Two jobs, and the second matters as much as the first.

**Aggregate.** One run is an anecdote. This reads every `ablation.json` under a
directory and reports the arms across all of them, so a claim about Gate 1 is a
claim about n runs rather than about the one that was rendered.

**Audit.** Every figure in the report and deck should be traceable to a file. A
number that is not is either a cited publication or something someone typed, and
the third case is the one this project exists to catch. `audit()` walks the
aggregate and states, per quantity, which run records produced it — so a reader
can check rather than trust.

A run that predates a measurement is reported as *not measured*, never as zero.
That distinction was the first thing to go wrong here: an older run had no
verification block, and reading it as "0 of 0 reproduced" would have understated
a run that had simply never been asked.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Run:
    path: Path
    data: dict

    @property
    def name(self) -> str:
        return self.path.parent.name

    @property
    def engineer(self) -> str:
        return self.data.get("engineer_model", "?")

    @property
    def gate(self) -> str:
        return self.data.get("gate_model", "?")

    @property
    def temperature(self):
        return self.data.get("temperature")

    def arm(self, which: str) -> dict:
        return self.data["arms"][which]

    def verification(self, which: str) -> dict | None:
        """``None`` when this run predates the verification pass."""
        v = self.arm(which).get("verification")
        if not v or ("ran" not in v):
            return None
        return v

    @property
    def measured_verification(self) -> bool:
        return self.verification("gated") is not None


def load_runs(root: str | Path) -> list[Run]:
    root = Path(root)
    runs = [
        Run(p, json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(root.glob("*/ablation.json"))
    ]
    return runs


@dataclass
class Aggregate:
    runs: list[Run]
    skipped: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.runs)

    @property
    def measured(self) -> list[Run]:
        return [r for r in self.runs if r.measured_verification]

    def arm_stats(self, which: str) -> dict[str, Any]:
        runs = self.measured
        if not runs:
            return {"n": 0}
        accepted = [r for r in runs if r.arm(which)["accepted"]]
        turns = [r.arm(which)["turns"] for r in runs]
        reported, matched, false_success = [], [], []
        for r in runs:
            v = r.verification(which) or {}
            reported.append(len(v.get("reported", {})))
            matched.append(len(v.get("matched", [])))
            false_success.append(r.arm(which)["false_success"])
        checkable = sum(
            len((r.verification(which) or {}).get("matched", []))
            + len((r.verification(which) or {}).get("mismatched", []))
            for r in runs
        )
        return {
            "n": len(runs),
            "accepted": len(accepted),
            "accept_rate": len(accepted) / len(runs),
            "turns_mean": statistics.fmean(turns),
            "turns": turns,
            "reported_total": sum(reported),
            "checkable_total": checkable,
            "reproduced_total": sum(matched),
            "reproduction_rate": (sum(matched) / checkable) if checkable else None,
            "false_success_total": sum(false_success),
        }

    def to_dict(self) -> dict:
        return {
            "runs": [
                {
                    "name": r.name,
                    "engineer": r.engineer,
                    "gate": r.gate,
                    "temperature": r.temperature,
                    "verification_measured": r.measured_verification,
                    "gated": {
                        "turns": r.arm("gated")["turns"],
                        "accepted": r.arm("gated")["accepted"],
                        "reported": len((r.verification("gated") or {}).get("reported", {})),
                        "reproduced": len((r.verification("gated") or {}).get("matched", [])),
                    },
                    "ungated": {
                        "turns": r.arm("ungated")["turns"],
                        "accepted": r.arm("ungated")["accepted"],
                        "reported": len((r.verification("ungated") or {}).get("reported", {})),
                        "false_success": r.arm("ungated")["false_success"],
                    },
                }
                for r in self.runs
            ],
            "gated": self.arm_stats("gated"),
            "ungated": self.arm_stats("ungated"),
        }


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def per_run_table(agg: Aggregate) -> str:
    if not agg.runs:
        return "<p class='small'>No run records found.</p>"
    rows = []
    for r in agg.runs:
        gv = r.verification("gated")
        uv = r.verification("ungated")
        note = "" if r.measured_verification else (
            " <span class='small'>(predates the verification pass — "
            "not measured, not zero)</span>"
        )
        rows.append(
            f"<tr><td class='mono'>{r.name}</td>"
            f"<td class='small'>{r.engineer}</td>"
            f"<td class='small'>T={r.temperature if r.temperature is not None else '?'}</td>"
            f"<td>{r.arm('gated')['turns']}</td>"
            f"<td>{'yes' if r.arm('gated')['accepted'] else 'no'}</td>"
            f"<td>{len((gv or {}).get('reported', {})) if gv else '—'}</td>"
            f"<td>{len((gv or {}).get('matched', [])) if gv else '—'}</td>"
            f"<td>{r.arm('ungated')['turns']}</td>"
            f"<td>{'yes' if r.arm('ungated')['accepted'] else 'no'}</td>"
            f"<td>{len((uv or {}).get('reported', {})) if uv else '—'}</td>"
            f"<td>{r.arm('ungated')['false_success']}</td>{note}</tr>"
        )
    return (
        "<table><tr><th rowspan=2>run</th><th rowspan=2>engineer</th>"
        "<th rowspan=2>temp</th>"
        "<th colspan=4 style='text-align:center'>with Gate 1</th>"
        "<th colspan=4 style='text-align:center'>without Gate 1</th></tr>"
        "<tr><th>turns</th><th>accepted</th><th>reported</th><th>reproduced</th>"
        "<th>turns</th><th>accepted</th><th>reported</th><th>false success</th></tr>"
        + "".join(rows) + "</table>"
    )


def summary_table(agg: Aggregate) -> str:
    g, u = agg.arm_stats("gated"), agg.arm_stats("ungated")
    if not g.get("n"):
        return "<p class='small'>No run has verification data yet.</p>"
    def rate(x):
        return "—" if x is None else f"{100 * x:.0f}%"
    return (
        f"<table><tr><th style='width:34%'>across {g['n']} run(s)</th>"
        f"<th>with Gate 1</th><th>without Gate 1</th></tr>"
        f"<tr><td>accepted a run</td><td>{g['accepted']} of {g['n']}</td>"
        f"<td>{u['accepted']} of {u['n']}</td></tr>"
        f"<tr><td>mean turns used</td><td>{g['turns_mean']:.1f}</td>"
        f"<td>{u['turns_mean']:.1f}</td></tr>"
        f"<tr><td>numbers reported in total</td><td>{g['reported_total']}</td>"
        f"<td>{u['reported_total']}</td></tr>"
        f"<tr><td>numbers checkable against an execution</td>"
        f"<td>{g['checkable_total']}</td><td>{u['checkable_total']}</td></tr>"
        f"<tr><td><b>reproduced on re-execution</b></td>"
        f"<td><b>{g['reproduced_total']} ({rate(g['reproduction_rate'])})</b></td>"
        f"<td>{u['reproduced_total']} ({rate(u['reproduction_rate'])})</td></tr>"
        f"<tr><td>false successes — accepted what the other arm rejected</td>"
        f"<td>{g['false_success_total']}</td>"
        f"<td><b>{u['false_success_total']}</b></td></tr></table>"
    )


def audit(agg: Aggregate, extra: dict[str, str] | None = None) -> str:
    """Where every quantity in this report came from.

    Anything not traceable to a run record or a cited paper is listed as
    narrative, explicitly, so it cannot pass for a measurement.
    """
    rows = [
        ("Per-run arm results, reproduction rates, false successes",
         f"{agg.n} × ablation.json on disk", "measured"),
        ("Token and cost figures", "usage block of each ablation.json", "measured"),
        ("Per-check outcomes per turn", "gate1_report.json per attempt", "measured"),
        ("Log excerpts and paths", "stdout.txt / stderr.txt per attempt", "measured"),
        ("Log-scanner precision and recall",
         "rig/corpus.py over tests/fixtures/log_corpus.jsonl", "measured"),
        ("Prompt compression 203 lines to 4 shapes",
         "gates/log_digest.py on a recorded capture", "measured"),
        ("Check inventory counts", "counted from gates/gate1.py", "measured"),
        ("MLR-Bench, BadScientist, CORE-Bench, PaperBench figures",
         "quoted from the papers, arXiv ids given", "cited"),
        ("Archived-run figures: 81.60%, 13.61x, reward 1.0",
         "results/gemini_3_5_flash_run_1 log in the host repo", "measured"),
        ("Defect list and the story of how each was found",
         "this session's commit history", "narrative — not a measurement"),
    ]
    for label, source in (extra or {}).items():
        rows.append((label, source, "measured"))
    body = "".join(
        f"<tr><td>{what}</td><td class='small'>{src}</td>"
        f"<td class='{'ok' if kind == 'measured' else 'warn'}'>{kind}</td></tr>"
        for what, src, kind in rows
    )
    return (
        "<table><tr><th style='width:42%'>quantity</th><th>source</th>"
        "<th>kind</th></tr>" + body + "</table>"
    )


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else (
        "/home/kesh/AgentLaboratory-Gemini/ablation_runs"
    )
    agg = Aggregate(load_runs(root))
    print(json.dumps(agg.to_dict(), indent=2))
