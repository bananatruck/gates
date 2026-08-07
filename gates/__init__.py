"""G.A.T.E.S. — a portable validity layer for autonomous research agents.

Three gates sit at the points where an autonomous research scaffold loses its
grip on evidence:

* **Gate 1 — execution validity.** Did the code actually run, and were the
  reported numbers produced by this run? Deterministic; no model consulted.
* **Gate 2 — source ↔ result coherence.** Are the measured results consistent
  with what the cited literature reports? (pending)
* **Gate 3 — report validity.** Does every number and citation in the manuscript
  trace to something that exists? (pending)

Nothing in this package imports a host scaffold. Porting to a new one means
writing a single adapter — see ``adapters/agentlab.py`` for the reference
implementation against Agent-Researcher / Agent Laboratory.
"""

from .errors import GateError, GateFailure, HarnessError
from .gate1 import GATE_NAME as GATE1_NAME, Gate1Config, run_gate1
from .ledger import Ledger
from .report import render_feedback, render_summary
from .runner import code_sha256, run_experiment
from .schema import (
    SCHEMA_VERSION,
    CheckResult,
    ExecutionRecord,
    GateReport,
    MetricRecord,
    Severity,
    Verdict,
)

__all__ = [
    "SCHEMA_VERSION",
    "CheckResult",
    "ExecutionRecord",
    "GATE1_NAME",
    "Gate1Config",
    "GateError",
    "GateFailure",
    "GateReport",
    "HarnessError",
    "Ledger",
    "MetricRecord",
    "Severity",
    "Verdict",
    "code_sha256",
    "render_feedback",
    "render_summary",
    "run_experiment",
    "run_gate1",
]
