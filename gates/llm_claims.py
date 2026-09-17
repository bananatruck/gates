"""Model-assisted claim scanning - the recall half of Gate 3's model layer (D31).

Gate 1's log scan reads the log lines its patterns did not flag. This reads the
findings prose the number scanner did not flag. ``prose.NUMBER`` needs a
decimal point or four digits, ``SKIP_LINE`` drops whole ``\\cite`` lines, and
``\\begin{abstract}`` is not a heading, so all of these pass it clean:

    accuracy improves by 9 points
    eighty-one percent of the test nodes
    SGC trains twice as fast

Each states a measured quantity with no ``\\result{}`` token. Reading that is
what a model is for.

**Findings are WARN, by construction.** They are built with
``llm.model_warning``, so a model can never fail a manuscript. A spelled-out
number still ships, with a warning the writer is sent and the report keeps.
This module never names a failing severity, and a test parses it to check.

**Findings are grounded.** The model names a row and quotes the words that
state the quantity. A row it was not shown, or a quote the row does not
contain, is dropped rather than repaired.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import ModelLayer, model_warning
from .llm_scan import _extract_json_array
from .prose import CLAIM_SECTIONS, flags_claim, sections
from .schema import CheckResult, Severity

CHECK_ID = "report.model_unbound_claims"

#: Rows sent per manuscript. Bounds cost, and is reported so the recall claim
#: is bounded by what was actually examined.
DEFAULT_MAX_LINES = 200

#: Findings kept. Beyond a handful the feedback stops being small.
MAX_FINDINGS = 6

SYSTEM = (
    "You audit the findings prose of a research paper before it is published. "
    "Every number that reports a result of this study must appear as the "
    "placeholder RESULT, which is later replaced by a measured value. Your job "
    "is to find rows that state a measured quantity WITHOUT a RESULT "
    "placeholder.\n\n"
    "Report a row if it states a quantity this study measured in any other "
    "form: a small integer ('improves by 9 points', '3 of 5 seeds'), a number "
    "in words ('eighty-one percent'), or a ratio or multiple ('twice as fast', "
    "'a third fewer errors').\n\n"
    "Do NOT report: the RESULT placeholder itself, citations, years, section or "
    "figure numbers, names that contain digits (CIFAR-10, ResNet-50, GPT-4), "
    "or numbers quoted from other papers. A false positive sends the writer to "
    "fix a sentence that was fine, so when unsure, leave it out.\n\n"
    "Respond with a JSON array and nothing else. Each element: "
    '{"line": <the row number as given>, "quote": "<the exact words in that row '
    'that state the quantity>", "why": "<at most 15 words>"}. '
    "If nothing qualifies, respond with []."
)

_ABSTRACT_ENV = re.compile(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", re.DOTALL)


@dataclass(frozen=True)
class ClaimFinding:
    section: str
    line: str
    quote: str
    why: str


@dataclass(frozen=True)
class ClaimScan:
    """What the model contributed, and how much of the prose it actually saw."""

    findings: list[ClaimFinding]
    lines_examined: int
    lines_skipped: int
    ok: bool
    error: str | None = None


def candidate_lines(masked: str) -> list[tuple[str, str]]:
    """Findings rows the number scanner did not flag, as ``(section, line)``.

    ``masked`` has its result tokens already replaced by ``RESULT``. Rows the
    scanner flagged are left out: re-sending them would inflate what the model
    appears to contribute, which is the number this layer has to report
    honestly.
    """
    bodies = [(h, b) for h, b in sections(masked) if any(s in h for s in CLAIM_SECTIONS)]
    bodies += [("abstract", m.group(1)) for m in _ABSTRACT_ENV.finditer(masked)]
    return [
        (heading, line.strip())
        for heading, body in bodies
        for line in body.splitlines()
        if line.strip() and not flags_claim(line)
    ]


def scan_claims(
    layer: ModelLayer, masked: str, *, max_lines: int = DEFAULT_MAX_LINES
) -> ClaimScan:
    """Ask the model about the findings rows the number scanner passed."""
    candidates = candidate_lines(masked)
    kept, skipped = candidates[:max_lines], max(0, len(candidates) - max_lines)
    if not kept:
        return ClaimScan([], 0, skipped, ok=True)

    numbered = "\n".join(
        f"{i}\t[{section}]\t{line}" for i, (section, line) in enumerate(kept, start=1)
    )
    call = layer.ask(
        "Findings prose, one row per line. The first column is the ROW NUMBER; "
        f"valid values are 1 to {len(kept)}. The second is the section.\n\n"
        f"{numbered}",
        SYSTEM,
    )
    if not call.ok:
        return ClaimScan([], len(kept), skipped, ok=False, error=call.error)
    return ClaimScan(_parse(call.text, kept), len(kept), skipped, ok=True)


def _parse(text: str, rows: list[tuple[str, str]]) -> list[ClaimFinding]:
    """Findings from the completion, with anything ungrounded dropped."""
    payload = _extract_json_array(text) or []
    findings: list[ClaimFinding] = []
    seen: set[int] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("line"))
        except (TypeError, ValueError):
            continue
        quote = str(item.get("quote") or "").strip()
        if not 1 <= index <= len(rows) or index in seen or not quote:
            continue
        section, line = rows[index - 1]
        if quote.lower() not in line.lower():
            continue
        seen.add(index)
        why = str(item.get("why") or "").strip()
        findings.append(ClaimFinding(section, line, quote, why))
        if len(findings) >= MAX_FINDINGS:
            break
    return findings


def build_check(scan: ClaimScan) -> CheckResult | None:
    """The row the gate appends, or ``None`` when there is nothing to say.

    A scan that found nothing is silence, not a passing row. A scan that could
    not run is recorded, because it bounds what the report may claim.
    """
    if not scan.ok:
        return CheckResult(
            id=CHECK_ID,
            passed=True,
            severity=Severity.INFO,
            message=(
                "the claim scan could not run, so numbers the pattern scanner "
                "cannot see (small integers, numbers in words) were not looked for"
            ),
            evidence={"error": scan.error, "degraded": True},
        )
    if not scan.findings:
        return None
    return model_warning(
        CHECK_ID,
        (
            f"{len(scan.findings)} findings row(s) state a measured quantity "
            f"without a result token"
        ),
        {
            "findings": [
                {"section": f.section, "quote": f.quote, "why": f.why, "line": f.line}
                for f in scan.findings
            ],
            "lines_examined": scan.lines_examined,
            "lines_skipped": scan.lines_skipped,
        },
    )
