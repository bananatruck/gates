"""Diagnostics over the captured logs.

Gate 1's hard checks answer "did the process fail". This module answers the
narrower question that survives a clean exit: *did the run report trouble in its
own output and carry on anyway?*

A run that exits 0 can still have printed a caught exception, silently produced a
NaN through a division by zero, or fallen back to CPU after a CUDA failure. None
of those break the process, so no exit code records them — but each one changes
what the recorded numbers mean, and the writing agent must not present them as a
clean result.

Findings here never block. They are surfaced so the report states them.

Patterns are chosen for precision over coverage, because a false positive costs
the agent a rewrite for nothing. In particular the exception-name pattern
requires CamelCase before the colon, so ``ValueError:`` matches while
``Mean Squared Error:`` — ordinary ML prose — does not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Owned by ``exec.no_swallowed_traceback``, so excluded from this table.
TRACEBACK_MARKER = "Traceback (most recent call last)"

_MAX_FINDINGS_PER_SIGNAL = 3
_MAX_LINE_CHARS = 200


@dataclass(frozen=True)
class LogSignal:
    id: str
    #: What it means for the recorded values, in the agent's terms.
    note: str
    pattern: re.Pattern[str]


@dataclass
class LogFinding:
    signal: str
    note: str
    stream: str  # "stdout" | "stderr"
    lineno: int
    line: str


def _p(expr: str) -> re.Pattern[str]:
    return re.compile(expr)


#: Specific diagnoses first, catch-alls last. Only the first match on a line is
#: kept, and "CUDA out of memory" is more use to the agent than "an exception was
#: printed" — even though the line satisfies both.
SIGNALS: tuple[LogSignal, ...] = (
    LogSignal(
        id="numerical_integrity",
        note=(
            "a numerical warning fired during the run — any metric downstream of "
            "it may be NaN, infinite, or silently wrong"
        ),
        pattern=_p(
            r"invalid value encountered"
            r"|divide by zero encountered"
            r"|overflow encountered"
            r"|underflow encountered"
            r"|Mean of empty slice"
            r"|Degrees of freedom <= 0"
        ),
    ),
    LogSignal(
        id="device_failure",
        note=(
            "a device or memory failure occurred — the run may have silently "
            "fallen back to a different device or a smaller batch"
        ),
        pattern=_p(
            r"CUDA out of memory"
            r"|CUDA error"
            r"|cuda runtime error"
            r"|OutOfMemoryError"
            r"|falling back to CPU"
        ),
    ),
    LogSignal(
        id="convergence",
        note=(
            "the fit did not converge, so the reported metric is not the "
            "converged one"
        ),
        pattern=_p(
            r"ConvergenceWarning"
            r"|did not converge"
            r"|failed to converge"
            r"|Maximum number of iterations reached"
        ),
    ),
    LogSignal(
        id="printed_exception",
        note=(
            "the run printed an exception and continued, so the surrounding "
            "numbers come from a partially failed execution"
        ),
        # CamelCase before the colon: ValueError: yes, "Squared Error:" no.
        pattern=_p(r"\b[A-Z][A-Za-z_]*(?:Error|Exception)\s*:"),
    ),
    LogSignal(
        id="logged_error_level",
        note="the run logged at ERROR level and continued",
        pattern=_p(r"(?:^|\s)(?:\[ERROR\]|ERROR:)|\s-\s+ERROR\s+-"),
    ),
)


def scan(text: str, stream: str) -> list[LogFinding]:
    """Findings in one captured stream, capped per signal."""
    findings: list[LogFinding] = []
    counts: dict[str, int] = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or TRACEBACK_MARKER in line:
            continue
        for signal in SIGNALS:
            if counts.get(signal.id, 0) >= _MAX_FINDINGS_PER_SIGNAL:
                continue
            if signal.pattern.search(line):
                counts[signal.id] = counts.get(signal.id, 0) + 1
                findings.append(
                    LogFinding(
                        signal=signal.id,
                        note=signal.note,
                        stream=stream,
                        lineno=lineno,
                        line=_clip(line),
                    )
                )
                break  # one finding per line; the first signal is the worst one
    return findings


def scan_streams(stdout: str, stderr: str) -> list[LogFinding]:
    return scan(stdout, "stdout") + scan(stderr, "stderr")


def count_tracebacks(stdout: str, stderr: str) -> tuple[int, int]:
    """Traceback occurrences per stream.

    Both streams matter: an agent that writes ``except Exception: print(e)`` or
    ``traceback.print_exc(file=sys.stdout)`` puts the evidence on stdout, where a
    stderr-only check never sees it.
    """
    return stdout.count(TRACEBACK_MARKER), stderr.count(TRACEBACK_MARKER)


def _clip(line: str) -> str:
    if len(line) <= _MAX_LINE_CHARS:
        return line
    return f"{line[:_MAX_LINE_CHARS]}… (+{len(line) - _MAX_LINE_CHARS} chars)"
