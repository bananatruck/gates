"""The verified value registry — the artifact Gate 3 is allowed to cite from.

Two documented requirements shape this file.

**"Typed and hashed to the run that produced it."** Every entry carries its unit,
its call site, and a ``trace_id`` derived from the run that recorded it. A number
in the registry is not a number that appeared somewhere; it is a number
attributable to one execution of one exact source text.

**An unbroken causal chain, not a value match.** A claim whose number happens to
match something in a log — but for which no recorded command produced that run —
is a worse sign than a numeric discrepancy, because it suggests the value was
backfilled. So each entry carries its chain explicitly, link by link
(task → command → log → value), with each link marked resolved or missing.
Gate 3 can then distinguish "traced" from "merely matches", and
``chain_integrity`` reports the rate rather than asserting a guarantee.

Only a registry whose gate verdict is PASS is citable. That is recorded in the
file itself so a consumer cannot forget to check.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .schema import SCHEMA_VERSION, GateReport

REGISTRY_FILENAME = "registry.json"

#: The provenance links Gate 1 can establish. "claim" is deliberately absent —
#: Gate 3 appends it when a claim in the manuscript resolves to a trace id.
CHAIN_LINKS = ("task", "command", "log", "value")


def build_registry(report: GateReport, *, task_ref: str | None = None) -> dict[str, Any]:
    """The registry for one gate report."""
    execution = report.execution
    parent_hash = report.code_sha256
    child_hash = execution.code_sha256 if execution else None
    # The parent hashes what it wrote; the child hashes what it ran. Equality is
    # what makes "this value came from this code" a checkable statement.
    hash_verified = bool(parent_hash) and parent_hash == child_hash

    run: dict[str, Any] = {
        "run_id": execution.run_id if execution else "",
        "code_sha256": parent_hash,
        "code_sha256_as_executed": child_hash,
        "code_sha256_verified": hash_verified,
        "argv": list(execution.argv) if execution else [],
        "started_at": execution.started_at if execution else "",
        "finished_at": execution.finished_at if execution else "",
        "duration_s": round(execution.duration_s, 3) if execution else None,
        "exit_code": execution.exit_code if execution else None,
        "artifact_dir": report.artifact_dir,
        "stdout_path": execution.stdout_path if execution else None,
        "stderr_path": execution.stderr_path if execution else None,
        "results_json_path": execution.results_json_path if execution else None,
        "seed": execution.seed() if execution else None,
        "metadata": dict(execution.metadata) if execution else {},
        "environment": dict(execution.environment) if execution else {},
    }

    task_hash = _digest(task_ref) if task_ref else None
    values: dict[str, Any] = {}
    for key, metric in (execution.metrics if execution else {}).items():
        chain = _chain(
            task_hash=task_hash,
            run=run,
            trace_id=metric.trace_id,
            hash_verified=hash_verified,
        )
        values[key] = {
            "trace_id": metric.trace_id,
            "value": metric.value,
            "unit": metric.unit,
            "type": type(metric.value).__name__,
            "provenance": {
                "run_id": run["run_id"],
                "code_sha256": parent_hash,
                "lineno": metric.lineno,
                "source_line": metric.source_line,
                "arg_kind": metric.arg_kind,
                "call_count": metric.call_count,
                "observations": list(metric.observations),
                "observations_truncated": metric.observations_truncated,
            },
            "chain": chain,
            "chain_complete": all(link["resolved"] for link in chain),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "gate": report.gate,
        "verdict": report.verdict.value,
        # A rejected run's values must never reach a manuscript. Recorded here so
        # a consumer that forgets to look at the verdict still cannot cite them.
        "citable": report.passed,
        "attempt": report.attempt,
        "rewrite": report.rewrite,
        "task_ref_sha256": task_hash,
        "run": run,
        "values": values,
        "chain_integrity": _integrity(values),
    }


def write_registry(
    report: GateReport,
    artifact_dir: str | Path,
    *,
    task_ref: str | None = None,
) -> Path:
    path = Path(artifact_dir) / REGISTRY_FILENAME
    registry = build_registry(report, task_ref=task_ref)
    path.write_text(json.dumps(registry, indent=2, default=str), encoding="utf-8")
    return path


def load_registry(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def citable_values(registry: dict[str, Any]) -> dict[str, Any]:
    """The values a manuscript may use: none, unless the gate passed."""
    if not registry.get("citable"):
        return {}
    return {k: v["value"] for k, v in (registry.get("values") or {}).items()}


def resolve_trace(registry: dict[str, Any], trace_id: str) -> dict[str, Any] | None:
    for key, entry in (registry.get("values") or {}).items():
        if entry.get("trace_id") == trace_id:
            return {"key": key, **entry}
    return None


def chain_integrity(registry: dict[str, Any]) -> dict[str, Any]:
    return registry.get("chain_integrity") or _integrity(registry.get("values") or {})


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _chain(
    *,
    task_hash: str | None,
    run: dict[str, Any],
    trace_id: str | None,
    hash_verified: bool,
) -> list[dict[str, Any]]:
    """One value's provenance, link by link.

    A missing link is reported rather than papered over: an unresolved chain is
    the signal that a number cannot be causally attributed, which is precisely
    the case that must outrank a numeric discrepancy.
    """
    log_path = run.get("stdout_path")
    return [
        {
            "link": "task",
            "ref": task_hash,
            "resolved": task_hash is not None,
            "why": None if task_hash else "the host scaffold supplied no task reference",
        },
        {
            "link": "command",
            "ref": run.get("run_id"),
            "resolved": bool(run.get("run_id")) and bool(run.get("argv")) and hash_verified,
            "why": None if hash_verified else "the executed source did not hash to the recorded source",
        },
        {
            "link": "log",
            "ref": log_path,
            "resolved": bool(log_path) and Path(log_path).exists(),
            "why": None if log_path else "no captured log for this run",
        },
        {
            "link": "value",
            "ref": trace_id,
            "resolved": bool(trace_id),
            "why": None if trace_id else "the value carries no trace id",
        },
    ]


def _integrity(values: dict[str, Any]) -> dict[str, Any]:
    """Causal-chain integrity rate: complete chains over all recorded values."""
    total = len(values)
    complete = sum(1 for v in values.values() if v.get("chain_complete"))
    broken: dict[str, list[str]] = {}
    for key, entry in values.items():
        missing = [link["link"] for link in entry.get("chain", []) if not link["resolved"]]
        if missing:
            broken[key] = missing
    return {
        "values": total,
        "complete_chains": complete,
        "rate": (complete / total) if total else None,
        "missing_links": broken,
    }


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
