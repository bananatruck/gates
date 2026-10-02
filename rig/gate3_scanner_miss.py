"""G3-M4: what the number scanner misses, measured on our own two manuscripts.

The plan asks for this and calls it the honest metric. A scanner with an
unmeasured false-negative rate is a scanner a reviewer will not trust, and
``report.no_numeric_literals_in_results`` is a scanner over prose, so the
"eliminated by construction" claim belongs to the *pipeline* and not to it.

**Measured, then fixed once (D38, D76).** The published Gate 1 traceability
number came from this scanner reading ``.tex`` (see ``gates/prose.py``), so
D38 froze it and D48 reported 34 of 49. D76 lifted the freeze for one rule,
masking references per token instead of skipping their lines, and the
measurement moved to 46 of 49. D104 reads the abstract environment and gives
each repeat of a value on one line its own context: 47 of 49, the two small
integers left, 13 false positives. Every remaining gap is reported and left
in place.

**Denominator (D39).** Two manuscripts, the gated and ungated arms of the
archived run. That is a small and non-independent sample, so this reports counts
and a raw fraction and no confidence interval: an interval over two documents
would claim a precision the sample does not have.

The labels are in ``rig/gate3_m4_labels.py`` and Kesh reviews them (D37).

Run: ``python -m rig.gate3_scanner_miss``
"""

from __future__ import annotations

import hashlib
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gates.prose import (  # noqa: E402
    claim_sections,
    findings_lines,
    line_claim_tokens,
)

from rig.gate3_m4_labels import (  # noqa: E402
    CAUSES,
    CLAIMS,
    FALSE_POSITIVES,
    MANUSCRIPT_SHA256,
    MISS_CAUSE,
)

PAPERS = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "finalized-report-and-results"
    / "verification"
    / "papers"
)

@dataclass
class Finding:
    """One labelled claim and whether the scanner reported it."""

    arm: str
    line: int
    token: str
    found: bool
    cause: str = ""


@dataclass
class ArmResult:
    arm: str
    #: Findings sections the scanner reached.
    sections_scanned: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    false_positives: list[tuple[int, str, str]] = field(default_factory=list)

    @property
    def claims(self) -> int:
        return len(self.findings)

    @property
    def detected(self) -> int:
        return sum(1 for f in self.findings if f.found)

    @property
    def missed(self) -> list[Finding]:
        return [f for f in self.findings if not f.found]


def _claim_lines(text: str) -> dict[int, str]:
    """Findings-section lines as ``{line: text}``, walked as ``extract_claims`` walks.

    The walk is ``prose.findings_lines`` itself, so this measurement and the
    scanner cannot drift apart. Since D104 it reads a ``\\begin{abstract}``
    block as the abstract; until then the ungated paper's abstract was
    reported here as a section the scanner never reached.
    """
    return {number: text for number, _, text in findings_lines(text)}


def _scanner_reports(line: str) -> Counter[str]:
    """The tokens ``extract_claims`` takes from this line, one per occurrence."""
    return Counter(token for token, _ in line_claim_tokens(line))


def measure(arm: str) -> ArmResult:
    """Score the scanner against the hand labels for one manuscript."""
    path = PAPERS / arm / "generated_report.txt"
    text = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != MANUSCRIPT_SHA256[arm]:
        raise SystemExit(
            f"{arm}: manuscript is not the one the labels were read against.\n"
            f"  labelled: {MANUSCRIPT_SHA256[arm]}\n"
            f"  on disk:  {digest}\n"
            "Line numbers in rig/gate3_m4_labels.py no longer apply. Re-label."
        )

    result = ArmResult(arm=arm)
    lines = _claim_lines(text)
    result.sections_scanned = sorted(set(claim_sections(text)))

    for line_no, tokens in sorted(CLAIMS[arm].items()):
        reported = _scanner_reports(lines[line_no])
        seen: Counter[str] = Counter()
        for token in tokens:
            seen[token] += 1
            found = seen[token] <= reported.get(token, 0)
            cause = "" if found else MISS_CAUSE[(arm, line_no)]
            result.findings.append(Finding(arm, line_no, token, found, cause))

    for line_no, rows in sorted(FALSE_POSITIVES.get(arm, {}).items()):
        for token, reason in rows:
            result.false_positives.append((line_no, token, reason))

    # Cross-check: every token the scanner reports is either a labelled claim or
    # a declared false positive. A third case means the labels are incomplete,
    # and absorbing it silently is the defect this project exists to catch.
    # Counted, not matched: a token reported twice on a line and labelled once
    # is one report nobody accounted for.
    unaccounted = sorted(
        (n, t)
        for n, line in lines.items()
        for t in (
            _scanner_reports(line)
            - Counter(CLAIMS[arm].get(n, ()))
            - Counter(t for t, _ in FALSE_POSITIVES.get(arm, {}).get(n, ()))
        ).elements()
    )
    if unaccounted:
        raise SystemExit(
            f"{arm}: the scanner reports {len(unaccounted)} token(s) that are "
            f"neither labelled a claim nor declared a false positive: "
            f"{unaccounted}"
        )
    return result


def main() -> int:
    arms = [measure("gated"), measure("ungated")]
    claims = sum(a.claims for a in arms)
    detected = sum(a.detected for a in arms)
    missed = [f for a in arms for f in a.missed]
    positives = detected + sum(len(a.false_positives) for a in arms)

    print("G3-M4 — what the number scanner misses")
    print(f"  corpus: {len(arms)} manuscripts, hand-labelled, no interval (D39)")
    print()
    for arm in arms:
        print(f"  {arm.arm:8}  {arm.detected:3} of {arm.claims:3} claims detected"
              f"   {len(arm.false_positives)} false positive(s)")
    print()
    print(f"  detected      {detected} of {claims}")
    print(f"  missed        {len(missed)} of {claims}"
          f"  ({len(missed) / claims:.1%} of labelled claims)")
    print(f"  reported      {positives}, of which {positives - detected} are not claims")
    print()
    print("  misses by cause")
    for cause, count in Counter(f.cause for f in missed).most_common():
        print(f"    {count:3}  {cause}")
        for line in _wrap(CAUSES[cause], 66):
            print(f"         {line}")
    print()
    print("  every missed claim")
    for f in missed:
        print(f"    {f.arm:8} line {f.line:4}  {f.token:22} {f.cause}")
    return 0


def _wrap(text: str, width: int) -> list[str]:
    out: list[str] = []
    line = ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
