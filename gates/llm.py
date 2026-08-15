"""The LLM layer — a required component of Gate 1, positioned outside the verdict.

Gate 1 needs a model for exactly two jobs, and neither is deciding whether a run
was valid:

1. **Log recall.** ``log_checks`` is a pattern set whose recall is openly
   unbounded; the model reads what the patterns did not flag.
2. **The feedback report.** The report is the loop's return path to the ML
   engineer, and "bind ``hidden_dim`` before use, or pass it into
   ``GCN.__init__``" is not something a template can write.

Where the layer sits is what makes both safe::

    static → execution → runtime → log → results checks
                                          ↓
                          LLM log scan  (WARN-severity only)
                                          ↓
                                     decide()      ← verdict fixed here, no LLM
                                          ↓
              PASS → evidence bundle        FAIL → LLM writes the feedback report

``decide()`` fails on blocking checks only, so a WARN-severity finding cannot
move a verdict however wrong it is; and report generation runs after the verdict
already exists. R1.9 — "gate verdict is computed with no LLM in the path" — holds
exactly as written.

The constraint is enforced structurally rather than by convention: model-derived
findings are constructed through :func:`model_warning`, which hardcodes
``Severity.WARN``. ``Severity.FAIL`` does not appear in this module, and a test
asserts that it never does.

The layer never raises into the gate. A model that times out, errors, or returns
nonsense degrades the report; it must not fail the run. A validity layer that
becomes unavailable when an API is down is not a validity layer.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable

from .schema import CheckResult, Severity

#: What a host adapter injects. ``(prompt, system_prompt) -> completion``.
#: Deliberately the narrowest signature that works: no client object, no
#: streaming, no tool use, nothing for ``gates/`` to depend on. Agent-Researcher
#: supplies a two-argument closure over ``inference.query_model``.
ModelFn = Callable[[str, str], str]

DEFAULT_TIMEOUT_S = 60.0

#: Prompts are truncated to this before sending. Captured stdout can be
#: megabytes; the layer must not turn a large experiment into a large bill.
DEFAULT_MAX_PROMPT_CHARS = 24_000


@dataclass
class ModelCall:
    """One invocation, and what happened to it.

    ``ok=False`` is an ordinary outcome, not an exception: every caller has a
    deterministic path to fall back to.
    """

    ok: bool
    text: str = ""
    error: str | None = None
    latency_s: float = 0.0
    prompt_chars: int = 0
    completion_chars: int = 0
    truncated_prompt: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error": self.error,
            "latency_s": round(self.latency_s, 3),
            "prompt_chars": self.prompt_chars,
            "completion_chars": self.completion_chars,
            "truncated_prompt": self.truncated_prompt,
        }


@dataclass
class ModelBudget:
    """What the layer spent on one attempt.

    Recorded per gate run and asserted against a ceiling in test, so a prompt
    change cannot quietly double the cost of running the gate.
    """

    calls: int = 0
    failures: int = 0
    latency_s: float = 0.0
    prompt_chars: int = 0
    completion_chars: int = 0
    errors: list[str] = field(default_factory=list)

    def record(self, call: ModelCall) -> None:
        self.calls += 1
        self.latency_s += call.latency_s
        self.prompt_chars += call.prompt_chars
        self.completion_chars += call.completion_chars
        if not call.ok:
            self.failures += 1
            if call.error:
                self.errors.append(call.error)

    @property
    def degraded(self) -> bool:
        """True when any call failed, so the report can say so rather than
        letting a degraded report pass for a complete one."""
        return self.failures > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "failures": self.failures,
            "degraded": self.degraded,
            "latency_s": round(self.latency_s, 3),
            "prompt_chars": self.prompt_chars,
            "completion_chars": self.completion_chars,
            "errors": self.errors[:5],
        }


class ModelLayer:
    """The injected callable, wrapped so the gate can depend on it not exploding.

    Responsibilities, all of them defensive: bound the prompt, bound the wait,
    swallow every exception into a ``ModelCall``, and keep the running cost.
    """

    def __init__(
        self,
        fn: ModelFn | None,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
    ):
        self.fn = fn
        self.timeout_s = timeout_s
        self.max_prompt_chars = max_prompt_chars
        self.budget = ModelBudget()

    @property
    def available(self) -> bool:
        return self.fn is not None

    def ask(self, prompt: str, system_prompt: str = "") -> ModelCall:
        """Call the model. Never raises."""
        if self.fn is None:
            call = ModelCall(ok=False, error="no model was supplied")
            self.budget.record(call)
            return call

        truncated = len(prompt) > self.max_prompt_chars
        if truncated:
            prompt = _clip_middle(prompt, self.max_prompt_chars)

        started = time.monotonic()
        try:
            text = self._call_with_deadline(prompt, system_prompt)
            call = ModelCall(
                ok=True,
                text=text if isinstance(text, str) else str(text),
                latency_s=time.monotonic() - started,
                prompt_chars=len(prompt),
                truncated_prompt=truncated,
            )
            call.completion_chars = len(call.text)
        except Exception as exc:  # noqa: BLE001 — any failure degrades, none blocks
            call = ModelCall(
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                latency_s=time.monotonic() - started,
                prompt_chars=len(prompt),
                truncated_prompt=truncated,
            )
        self.budget.record(call)
        return call

    def _call_with_deadline(self, prompt: str, system_prompt: str) -> str:
        """Best-effort deadline around an arbitrary callable.

        A thread cannot be killed, so on timeout the underlying request may
        still be in flight — the same limitation that made the host scaffold's
        ``future.result(timeout=)`` unable to stop a runaway experiment. Stated
        rather than hidden. The consequence here is bounded and different in
        kind: the gate proceeds with a degraded report while a network call
        finishes unobserved, rather than a training run burning CPU for the rest
        of the session. Hosts whose client supports a real timeout should set
        one; this is a floor, not a substitute.
        """
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(self.fn, prompt, system_prompt)
            try:
                return future.result(timeout=self.timeout_s)
            except FutureTimeout:
                raise TimeoutError(
                    f"model did not respond within {self.timeout_s:g}s"
                ) from None
        finally:
            # wait=False is load-bearing. Using the executor as a context
            # manager calls shutdown(wait=True), which blocks until the very
            # call we just gave up on returns — making the deadline decorative.
            # The worker is abandoned rather than joined.
            pool.shutdown(wait=False)


def model_warning(
    check_id: str,
    message: str,
    evidence: dict[str, Any] | None = None,
) -> CheckResult:
    """The only constructor available to model-derived findings.

    Severity is not a parameter. This is the whole guarantee that lets a
    required LLM coexist with a deterministic verdict: ``decide()`` fails on
    blocking checks, ``blocking`` requires ``Severity.FAIL``, and nothing the
    model produces can carry it.
    """
    return CheckResult(
        id=check_id,
        passed=False,
        severity=Severity.WARN,
        message=message,
        evidence=evidence or {},
    )


def _clip_middle(text: str, budget: int) -> str:
    """Keep the head and the tail; say exactly how much was dropped.

    The head carries setup and the tail carries the failure, so the middle is
    what an error signal is least likely to be hiding in.
    """
    if len(text) <= budget:
        return text
    half = budget // 2
    dropped = len(text) - budget
    return (
        f"{text[:half]}\n"
        f"[... {dropped:,} characters elided from the middle ...]\n"
        f"{text[-half:]}"
    )
