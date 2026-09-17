"""G3-M4: what the number scanner misses, measured on our own two manuscripts.

The plan asks for this and calls it the honest metric. A scanner with an
unmeasured false-negative rate is a scanner a reviewer will not trust, and
``report.no_numeric_literals_in_results`` is a scanner over prose, so the
"eliminated by construction" claim belongs to the *pipeline* and not to it.

**This measures and does not fix (D38).** The published Gate 1 traceability
number came from this scanner reading ``.tex`` (see ``gates/prose.py``), so
changing it restates a measured result. Every gap below is reported and left in
place.

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
    CITATION,
    CLAIM_SECTIONS,
    NUMBER,
    SKIP_LINE,
    _heading,
    is_claim,
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

#: LaTeX's abstract environment, which ``_heading`` does not see (D40). Tracked
#: here so a manuscript using it is not silently credited with no abstract: the
#: ungated paper's abstract is invisible to the scanner, and this measurement is
#: where that gap gets reported rather than closed.
_ABSTRACT_BEGIN = "\\begin{abstract}"
_ABSTRACT_END = "\\end{abstract}"


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
    #: Findings sections the scanner reached, and the ones it could not.
    sections_scanned: list[str] = field(default_factory=list)
    sections_invisible: list[str] = field(default_factory=list)
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


def _claim_lines(text: str) -> dict[int, tuple[str, bool]]:
    """Findings-section lines as ``{line: (text, visible)}``.

    Mirrors ``extract_claims``'s walk, with one deliberate difference: a
    ``\\begin{abstract}`` block counts as the abstract and is marked
    ``visible=False``. ``_heading`` does not see that environment (D40), so the
    scanner never reaches those lines at all. Treating the section as absent
    would hide the gap; marking it unreadable measures it.
    """
    out: dict[int, tuple[str, bool]] = {}
    section = "preamble"
    in_abstract = False
    for number, line in enumerate(text.splitlines(), 1):
        heading = _heading(line)
        if heading is not None:
            section = heading
            continue
        if _ABSTRACT_BEGIN in line:
            in_abstract = True
            continue
        if _ABSTRACT_END in line:
            in_abstract = False
            continue
        current = "abstract" if in_abstract else section
        if any(s in current for s in CLAIM_SECTIONS):
            out[number] = (line, not in_abstract)
    return out


def _scanner_reports(line: str, visible: bool = True) -> Counter[str]:
    """The tokens ``extract_claims`` would take from this line, as it takes them.

    Rebuilt rather than called, because ``extract_claims`` returns floats with no
    line numbers and this measurement needs to know which line a finding came
    from. The steps below are its steps, in its order.
    """
    if not visible or SKIP_LINE.search(line):
        return Counter()
    stripped = CITATION.sub(" ", line)
    kept = [t for t in NUMBER.findall(stripped) if is_claim(t)]
    # context_of() locates a token with line.find(), so repeats of one token on
    # one line share a context and all but the first are deduplicated away.
    return Counter(dict.fromkeys(kept, 1))


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
    unreadable = sorted(n for n, (_, visible) in lines.items() if not visible)
    if unreadable:
        result.sections_invisible.append(
            f"abstract, written as \\begin{{abstract}}: "
            f"{len(unreadable)} findings line(s) the scanner never reaches"
        )
    result.sections_scanned = sorted(
        {s for s in CLAIM_SECTIONS if any(
            s in (_heading(line) or "") for line in text.splitlines()
        )}
    )

    for line_no, tokens in sorted(CLAIMS[arm].items()):
        line, visible = lines[line_no]
        reported = _scanner_reports(line, visible)
        seen: Counter[str] = Counter()
        for token in tokens:
            seen[token] += 1
            found = seen[token] <= reported.get(token, 0)
            # Invisibility outranks the hand label: if the scanner cannot reach
            # the line, no property of the token explains the miss.
            cause = "" if found else (
                "invisible_section" if not visible else MISS_CAUSE[(arm, line_no)]
            )
            result.findings.append(Finding(arm, line_no, token, found, cause))

    for line_no, rows in sorted(FALSE_POSITIVES.get(arm, {}).items()):
        for token, reason in rows:
            result.false_positives.append((line_no, token, reason))

    # Cross-check: every token the scanner reports is either a labelled claim or
    # a declared false positive. A third case means the labels are incomplete,
    # and absorbing it silently is the defect this project exists to catch.
    labelled = {(n, t) for n, ts in CLAIMS[arm].items() for t in ts}
    declared = {(n, t) for n, rows in FALSE_POSITIVES.get(arm, {}).items() for t, _ in rows}
    unaccounted = sorted(
        (n, t)
        for n, (line, visible) in lines.items()
        for t in _scanner_reports(line, visible)
        if (n, t) not in labelled and (n, t) not in declared
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
        for note in arm.sections_invisible:
            print(f"            findings section the scanner cannot read: {note}")
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
