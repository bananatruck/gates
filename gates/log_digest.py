"""Collapse a log to its distinct shapes before anyone reads it.

Experiment logs are overwhelmingly repetitive. A 200-epoch training loop emits
200 lines carrying one fact; measured on a realistic capture, 203 non-blank lines
reduced to **four** distinct shapes. Sending all 203 to a model costs 14,000
characters to convey four things, and buries the two lines that matter in a wall
of epochs.

The collapse is *lossless in distinct content*, and that property is what makes
it safe here. Every distinct shape survives with its first real line number, so
nothing a scanner could have flagged disappears — the recall the model tier
exists to raise is not quietly traded for the cost saving. What is discarded is
only the 199th restatement of a shape already present.

Two details do real work:

``nan`` and ``inf`` are not digits, so ``loss nan`` normalises to a different
shape than ``loss 1.9243`` and survives as its own entry rather than being
absorbed into the epoch group. That is the single most important line in a
diverged run.

When a shape's members differ numerically, the last member is kept alongside the
first. A run whose accuracy drifts from 0.41 to 0.02 has two lines worth seeing,
and only the pair shows the drift.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Order matters: timestamps and hex before bare numbers, or the number rule
#: shreds them into fragments that no longer group.
_NORMALISERS: tuple[tuple[re.Pattern[str], str], ...] = (
    # 2026-07-22 21:45:23.509775 / 2026-07-22T21:45:23
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?"), "<ts>"),
    (re.compile(r"\b\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b"), "<time>"),
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<addr>"),
    (re.compile(r"\b[0-9a-f]{12,}\b"), "<hash>"),
    # Progress bars: 45%|####      | 90/200 [00:04<00:05, 45.21it/s]
    (re.compile(r"[|#█▉▊▋▌▍▎▏ ]{4,}"), "<bar>"),
    (re.compile(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?"), "<n>"),
)


@dataclass(frozen=True)
class DigestLine:
    """One distinct shape, and how many lines shared it."""

    stream: str
    #: Line number of the first member. Grounding resolves against this, so a
    #: model reporting a digested line still points at a real line in the file.
    lineno: int
    line: str
    count: int = 1
    #: The last member, when it differs textually from the first. Present only
    #: for groups whose values drifted.
    variant: str | None = None
    variant_lineno: int | None = None

    @property
    def collapsed(self) -> bool:
        return self.count > 1

    def render(self) -> str:
        if not self.collapsed:
            return self.line
        text = f"{self.line}    [x{self.count} identical in shape"
        if self.variant is not None:
            text += f"; last of them, line {self.variant_lineno}: {self.variant}"
        return text + "]"


def shape(line: str) -> str:
    """The line with its varying parts replaced, for grouping."""
    out = line
    for pattern, placeholder in _NORMALISERS:
        out = pattern.sub(placeholder, out)
    return out.strip()


def digest(
    candidates: list[tuple[str, int, str]], *, keep_variants: bool = True
) -> list[DigestLine]:
    """Group ``(stream, lineno, line)`` by shape, preserving first appearance.

    Ordering is by first occurrence rather than by frequency, because a log
    reads chronologically and a reader — human or model — uses that to tell
    setup from failure.
    """
    groups: dict[tuple[str, str], list[tuple[int, str]]] = {}
    order: list[tuple[str, str]] = []

    for stream, lineno, line in candidates:
        key = (stream, shape(line))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((lineno, line))

    digested: list[DigestLine] = []
    for key in order:
        stream, _ = key
        members = groups[key]
        first_lineno, first_line = members[0]
        variant = variant_lineno = None
        if keep_variants and len(members) > 1:
            last_lineno, last_line = members[-1]
            if last_line != first_line:
                variant, variant_lineno = last_line, last_lineno
        digested.append(
            DigestLine(
                stream=stream,
                lineno=first_lineno,
                line=first_line,
                count=len(members),
                variant=variant,
                variant_lineno=variant_lineno,
            )
        )
    return digested


def compression(candidates: list, digested: list) -> float:
    """Lines removed as a fraction of lines in. Reported, not assumed."""
    if not candidates:
        return 0.0
    return 1.0 - (len(digested) / len(candidates))
