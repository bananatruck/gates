"""Typed contract shared by the gates, the execution harness, and the adapters.

Nothing here imports the host scaffold. The harness (which runs in a separate
interpreter) deliberately does *not* import this module either — it emits plain
JSON matching ``SCHEMA_VERSION``, and this module parses it back.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Iterable

SCHEMA_VERSION = "1.1"

#: Names the harness injects into the experiment's global namespace. Anything
#: else present before execution means the namespace was not clean.
HARNESS_INJECTED_NAMES = frozenset(
    {
        "__name__",
        "__file__",
        "__doc__",
        "__builtins__",
        "__package__",
        "__loader__",
        "__spec__",
        "record_result",
        "record_metadata",
    }
)


class Severity(str, Enum):
    FAIL = "FAIL"
    WARN = "WARN"
    INFO = "INFO"


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass
class CheckResult:
    """Outcome of a single named check."""

    id: str
    passed: bool
    severity: Severity
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def blocking(self) -> bool:
        return not self.passed and self.severity is Severity.FAIL

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


@dataclass
class MetricRecord:
    """One value the experiment declared through ``record_result``.

    ``arg_kind`` is filled in by static analysis of the call site, not by the
    harness: a value that appears as a source literal was typed, not measured.

    ``trace_id`` binds the value to one execution. It is what separates a claim
    that traces back through an unbroken chain from a claim whose number merely
    matches something that appears in a log.
    """

    key: str
    value: Any
    unit: str | None = None
    lineno: int | None = None
    source_line: str = ""
    call_count: int = 1
    arg_kind: str = "unknown"  # "computed" | "literal" | "unknown"
    #: Every ``record_result`` call for this key, oldest first, capped by the
    #: harness. Retained because the last value is not necessarily the reported
    #: one.
    observations: list[dict[str, Any]] = field(default_factory=list)
    observations_truncated: bool = False
    trace_id: str | None = None

    @property
    def is_finite(self) -> bool:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            return True
        return math.isfinite(self.value)

    def numeric_observations(self) -> list[float]:
        out = []
        for obs in self.observations:
            v = obs.get("value")
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if math.isfinite(v):
                out.append(float(v))
        return out

    @property
    def varied(self) -> bool:
        """Recorded more than once, with the value changing between calls."""
        seen = [obs.get("value") for obs in self.observations]
        return len(seen) > 1 and len(set(map(repr, seen))) > 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExceptionRecord:
    type: str
    message: str
    traceback: str = ""
    filename: str | None = None
    lineno: int | None = None
    function: str | None = None
    source_line: str = ""

    def where(self) -> str:
        if self.filename and self.lineno:
            loc = f"{self.filename}:{self.lineno}"
            return f"{loc} in {self.function}" if self.function else loc
        return "<unknown location>"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionRecord:
    """Everything the runtime observed about one execution.

    This replaces the scaffold's 1000-character stdout prefix. ``stdout_path``
    points at the complete, untruncated capture.
    """

    exit_code: int | None
    timed_out: bool
    duration_s: float
    stdout_path: str
    stderr_path: str
    stdout_bytes: int
    stderr_bytes: int
    truncated: bool = False
    exception: ExceptionRecord | None = None
    metrics: dict[str, MetricRecord] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    initial_namespace: list[str] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    results_json_path: str | None = None
    harness_error: str | None = None
    #: Identifier for this execution. Every recorded value derives its trace id
    #: from it, so a value can be attributed to one run and no other.
    run_id: str = ""
    #: The exact command the runner issued — the "executed command" link of the
    #: provenance chain.
    argv: list[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    #: sha256 of the source as read by the child process that ran it.
    code_sha256: str | None = None

    @property
    def clean_exit(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and self.exception is None

    def seed(self) -> Any:
        """The seed the experiment declared, if it declared one."""
        for name in ("seed", "random_seed", "torch_seed", "numpy_seed"):
            if name in self.metadata:
                return self.metadata[name]
        for key, value in self.metadata.items():
            if str(key).lower().endswith("seed"):
                return value
        return None

    def stdout_text(self, limit: int | None = None) -> str:
        return _read(self.stdout_path, limit)

    def stderr_text(self, limit: int | None = None) -> str:
        return _read(self.stderr_path, limit)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "argv": self.argv,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "code_sha256": self.code_sha256,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "duration_s": round(self.duration_s, 3),
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "stdout_bytes": self.stdout_bytes,
            "stderr_bytes": self.stderr_bytes,
            "truncated": self.truncated,
            "exception": self.exception.to_dict() if self.exception else None,
            "metrics": {k: m.to_dict() for k, m in self.metrics.items()},
            "metadata": self.metadata,
            "initial_namespace": self.initial_namespace,
            "environment": self.environment,
            "results_json_path": self.results_json_path,
            "harness_error": self.harness_error,
        }


@dataclass
class GateReport:
    """A gate's verdict on one attempt, plus the evidence behind it."""

    gate: str
    verdict: Verdict
    #: Execution ordinal — one per subprocess run, and the artifact directory name.
    attempt: int
    max_attempts: int
    #: Which agent-level rewrite this execution belongs to. One rewrite may spend
    #: several executions on automated repair, and only rewrites are budgeted.
    rewrite: int = 0
    checks: list[CheckResult] = field(default_factory=list)
    execution: ExecutionRecord | None = None
    artifact_dir: str | None = None
    code_sha256: str | None = None
    #: What the LLM layer spent and whether any call failed. ``None`` when no
    #: model was supplied. Never influences ``verdict`` — see ``llm.py``.
    model: dict[str, Any] | None = None

    @property
    def passed(self) -> bool:
        return self.verdict is Verdict.PASS

    @property
    def model_degraded(self) -> bool:
        """A model call failed, so anything the layer produced is incomplete.

        Callers surface this rather than letting a degraded report pass for a
        full one.
        """
        return bool(self.model and self.model.get("degraded"))

    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if c.blocking]

    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity is Severity.WARN]

    def metrics(self) -> dict[str, MetricRecord]:
        return self.execution.metrics if self.execution else {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "gate": self.gate,
            "verdict": self.verdict.value,
            "attempt": self.attempt,
            "rewrite": self.rewrite,
            "max_attempts": self.max_attempts,
            "code_sha256": self.code_sha256,
            "artifact_dir": self.artifact_dir,
            "checks": [c.to_dict() for c in self.checks],
            "execution": self.execution.to_dict() if self.execution else None,
            "model": self.model,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


def decide(checks: Iterable[CheckResult]) -> Verdict:
    """A gate fails if any blocking check failed. Warnings never block."""
    return Verdict.FAIL if any(c.blocking for c in checks) else Verdict.PASS


def _read(path: str | None, limit: int | None) -> str:
    if not path:
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read() if limit is None else f.read(limit)
    except OSError:
        return ""
