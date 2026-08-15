"""The upstream channel, reconstructed — so the divergence is measured, not told.

Agent Laboratory decided whether an experiment worked by searching the returned
stdout for a marker string::

    if "[CODE EXECUTION ERROR]" in code_ret: return False, ...

and the marker was appended to the capture buffer *after* the program's own
output, which was then sliced ``[:1000]``. That test is mechanical, so it can be
reproduced exactly, with no model and no API key. What cannot be reproduced is
``get_score`` — an LLM at temperature 0.6 — so this module does not pretend to.

The split matters for what may be claimed:

* **detector arm** — exact. "Would upstream have noticed this run failed?" is a
  substring search over a reconstructed buffer, and the answer is deterministic.
* **reward arm** — needs a real model. ``run_loop`` accepts one; without it the
  ledger records ``reward_score: null`` rather than a plausible-looking number.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gates import ExecutionRecord, run_experiment

#: Upstream's ``execute_code(code_str, timeout=60, MAX_LEN=1000)``.
LEGACY_MAX_LEN = 1000

#: Upstream's crash marker, and the whole of its failure detection.
LEGACY_MARKER = "[CODE EXECUTION ERROR]"


def legacy_buffer(execution: ExecutionRecord) -> str:
    """The capture buffer as upstream built it, before the slice.

    Order is the defect: the marker goes on the end, after everything the
    program printed.
    """
    buffer = execution.stdout_text()
    if execution.exception is not None:
        buffer += f"{LEGACY_MARKER}: {execution.exception.message}\n"
        buffer += execution.exception.traceback or ""
    return buffer


def legacy_view(execution: ExecutionRecord, max_len: int = LEGACY_MAX_LEN) -> str:
    """What the solver and the writing agent actually received."""
    return legacy_buffer(execution)[:max_len]


def upstream_detects_failure(
    execution: ExecutionRecord, max_len: int = LEGACY_MAX_LEN
) -> bool:
    """Upstream's entire success test, reproduced verbatim."""
    return LEGACY_MARKER in legacy_view(execution, max_len)


@dataclass(frozen=True)
class ChannelPoint:
    """One width of the evidence channel, and what survives it."""

    max_len: int
    marker_visible: bool
    chars_delivered: int
    chars_lost: int


def channel_sweep(
    execution: ExecutionRecord,
    widths: tuple[int, ...] = (500, 1_000, 2_000, 4_000, 8_000, 16_000, 64_000),
) -> list[ChannelPoint]:
    """Marker visibility as a function of channel width, for one execution.

    The instrument for PLAN.md step 5 at the level Gate 1 can settle on its own:
    the width at which upstream's detector stops working is a property of the
    capture, not of any model. The fabrication-rate arm of that experiment still
    needs real runs and a real writer.
    """
    buffer = legacy_buffer(execution)
    total = len(buffer)
    points = []
    for width in widths:
        delivered = min(width, total)
        points.append(
            ChannelPoint(
                max_len=width,
                marker_visible=LEGACY_MARKER in buffer[:width],
                chars_delivered=delivered,
                chars_lost=total - delivered,
            )
        )
    return points


def marker_survives_at(execution: ExecutionRecord) -> int | None:
    """Smallest channel width at which upstream would have seen the crash.

    ``None`` when the run did not crash. For a crash, this is the number the
    channel-fidelity argument turns on: upstream's ceiling was 1,000.
    """
    buffer = legacy_buffer(execution)
    index = buffer.find(LEGACY_MARKER)
    if index < 0:
        return None
    return index + len(LEGACY_MARKER)


def shadow_execute(
    source: str,
    artifact_dir: str | Path,
    *,
    timeout_s: int = 120,
    cwd: str | None = None,
) -> ExecutionRecord:
    """Run code the gate rejected *before* execution, to see what upstream saw.

    Gate 1's static tier is the point — a program with an unbound name never
    costs a training run. But upstream had no static tier, so the honest
    counterfactual for a statically-rejected attempt is the run upstream would
    have paid for. Without this, those attempts have no legacy view at all and
    scoring them would be invention.
    """
    return run_experiment(source, artifact_dir, timeout_s=timeout_s, cwd=cwd)
