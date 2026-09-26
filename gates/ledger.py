"""Append-only record of what each gate decided, and what the reward model said.

The disagreement between the two is a headline result, not a debugging aid: the
archived run in this repository has a reward model scoring 0.95 → 0.98 → 1.0 on
an experiment that raised ``NameError`` on every attempt. This file is what turns
that anecdote into a rate.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .gate2 import GATE_NAME as GATE2_NAME
from .schema import GateReport


class Ledger:
    """JSONL sink. One line per attempt, written as it happens."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record_attempt(
        self,
        report: GateReport,
        *,
        phase: str,
        reward_score: float | None = None,
        reward_model: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = {
            "ts": time.time(),
            "phase": phase,
            "gate": report.gate,
            "attempt": report.attempt,
            "verdict": report.verdict.value,
            "failed_checks": [c.id for c in report.failed_checks()],
            "warned_checks": [c.id for c in report.warnings()],
            "code_sha256": report.code_sha256,
            "reward_score": reward_score,
            "reward_model": reward_model,
            "metrics": {k: m.value for k, m in report.metrics().items()},
            "artifact_dir": report.artifact_dir,
        }
        if report.execution is not None:
            row["exit_code"] = report.execution.exit_code
            row["stdout_bytes"] = report.execution.stdout_bytes
            row["duration_s"] = round(report.execution.duration_s, 3)
        if extra:
            row.update(extra)
        self._append(row)
        return row

    def _append(self, row: dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return out

    def divergence_summary(self) -> dict[str, Any]:
        """How often the reward model endorsed something the gate rejected."""
        rows = [r for r in self.rows() if r.get("reward_score") is not None]
        rejected = [r for r in rows if r["verdict"] == "FAIL"]
        endorsed = [r for r in rejected if float(r["reward_score"]) >= 0.9]
        return {
            "attempts_scored": len(rows),
            "gate_rejected": len(rejected),
            "gate_rejected_but_reward_ge_0.9": len(endorsed),
            "max_reward_on_rejected": max(
                (float(r["reward_score"]) for r in rejected), default=None
            ),
        }

    def loop_summary(self) -> dict[str, Any]:
        """Gate 2 loop behaviour (spec M5), from rows ``review_loop`` wrote.

        A run starts at ``turn`` 0. It enters the loop only if its first review
        failed; a run admitted first time is counted in ``runs_reviewed`` and
        nowhere else. ``unresolved_declared`` is Gate 2 working as designed: the
        budget was spent and the run proceeded with its discrepancies declared.

        Gate 2 rows only. Gate 3's loop writes ``turn`` to the same ledger, and
        counting its manuscripts here would restate Gate 2's published number.
        """
        # limit: runs are split on turn 0, so two loops appending to one
        # ledger at the same time would merge; key rows by run id if hosts ever
        # review concurrently.
        runs: list[list[dict[str, Any]]] = []
        for row in self.rows():
            if row.get("gate") != GATE2_NAME:
                continue
            if "turn" not in row or "max_attempts" not in row:
                continue
            if row["turn"] == 0 or not runs:
                runs.append([])
            runs[-1].append(row)
        entered = [r for r in runs if r[0]["verdict"] == "FAIL"]
        resolved = [r for r in entered if r[-1]["verdict"] != "FAIL"]
        unresolved = [r for r in entered if r[-1]["verdict"] == "FAIL"]
        declared = [r for r in unresolved if len(r) >= r[-1]["max_attempts"]]
        return {
            "runs_reviewed": len(runs),
            "runs_entering_loop": len(entered),
            "resolved_within_budget": len(resolved),
            "resolution_rate": len(resolved) / len(entered) if entered else None,
            "mean_attempts": (
                sum(len(r) for r in resolved) / len(resolved) if resolved else None
            ),
            "unresolved_declared": len(declared),
            "abandoned": len(unresolved) - len(declared),
        }
