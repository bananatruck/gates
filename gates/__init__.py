"""G.A.T.E.S. — a portable validity layer for autonomous research agents.

Three gates sit at the points where an autonomous research scaffold loses its
grip on evidence:

* **Gate 1 — execution validity.** Did the code actually run, and were the
  reported numbers produced by this run? Deterministic; no model consulted.
* **Gate 2 — source ↔ result coherence.** Is a verified run consistent with
  what its plan declared? Each tier runs only on what the host declares: tier A
  checks declared ranges and relations, tier B checks declared methodology
  fields, and tier C is the feedback loop. No model reaches a verdict (D19).
* **Gate 3 — report validity.** Does every number in the manuscript trace to
  something that was measured, and does every citation resolve to a paper the
  run retrieved? The writer emits ``\\result{key}`` tokens and the renderer, not
  the model, writes the digits.

``GATES_LEVEL`` (0 to 3) picks how many gates run, cumulatively; see
``pipeline.gate_level``.

Nothing in this package imports a host scaffold. The loops every host shares
are in ``pipeline.py``; porting to a new host means writing one small adapter,
with ``adapters/agentlab.py`` as the reference against Agent Laboratory.
"""

from .errors import GateError, GateFailure, HarnessError
from .gate1 import GATE_NAME as GATE1_NAME, Gate1Config, run_gate1
from .gate2 import (
    GATE_NAME as GATE2_NAME,
    Band,
    Gate2Config,
    Range,
    Relation,
    SourceClaim,
    band_for,
    run_gate2,
    unresolved_discrepancies,
)
from .gate3 import (
    GATE_NAME as GATE3_NAME,
    Gate3Config,
    Substitution,
    render_result_tokens,
    run_gate3,
)
from .ledger import Ledger
from .llm import ModelBudget, ModelCall, ModelFn, ModelLayer, model_warning
from .registry import (
    REGISTRY_FILENAME,
    build_registry,
    chain_integrity,
    citable_values,
    load_registry,
    resolve_trace,
    write_registry,
)
from .report import render_feedback, render_summary
from .runner import code_sha256, make_run_id, make_trace_id, run_experiment
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
    "Band",
    "CheckResult",
    "ExecutionRecord",
    "GATE1_NAME",
    "GATE2_NAME",
    "Gate1Config",
    "Gate2Config",
    "GateError",
    "GateFailure",
    "GateReport",
    "HarnessError",
    "Ledger",
    "MetricRecord",
    "ModelBudget",
    "ModelCall",
    "ModelFn",
    "ModelLayer",
    "REGISTRY_FILENAME",
    "Range",
    "Relation",
    "SCHEMA_VERSION",
    "Severity",
    "SourceClaim",
    "Verdict",
    "band_for",
    "build_registry",
    "chain_integrity",
    "citable_values",
    "code_sha256",
    "load_registry",
    "make_run_id",
    "make_trace_id",
    "model_warning",
    "render_feedback",
    "render_summary",
    "resolve_trace",
    "run_experiment",
    "run_gate1",
    "run_gate2",
    "GATE3_NAME",
    "Gate3Config",
    "Substitution",
    "render_result_tokens",
    "run_gate3",
    "unresolved_discrepancies",
    "write_registry",
]
