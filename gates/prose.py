"""Numeric claims in a manuscript, found the way a reader finds them.

Gate 3 needs to know which numbers a paper asserts as its own findings. That is
a different question from Gate 1's, and none of ``static_checks`` transfers:
``_is_literal_derived`` walks an ``ast.expr``, and prose has no AST. So this is
a scanner over lines and headings.

It was written for ``rig/paper_audit.py`` and lives here now because
``pyproject.toml`` packages ``gates*`` only, so ``gates`` cannot import ``rig``.
The rig imports it back and keeps its measurement code where it was.

**The heading bug this move fixes.** The original matched ``\\section{...}`` and
nothing else. The real generated manuscripts in ``reports/`` are Markdown and
use ``## Key Results``, so against them the section never left ``preamble``, no
claims were found, and the audit came back clean. A scanner that reports zero
findings because it could not read the file is the exact defect this project
exists to catch, so :func:`claim_sections` exists to make coverage visible: an
audit over zero sections must never be presented as an audit that found nothing.

LaTeX behaviour is deliberately unchanged. The published Gate 1 traceability
number came from this scanner reading ``.tex``, and altering that path would
silently restate a measured result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Percentages and decimals as a paper states them: "81.60\%", "$13.61\times$",
#: "0.0180 seconds". Bare small integers are excluded by :func:`is_claim`.
NUMBER = re.compile(r"(\d+\.\d+|\d{2,})")

#: Sections where a paper states its findings. Background and related work quote
#: other people's numbers, which are not this run's to source.
CLAIM_SECTIONS = (
    "abstract",
    "contributions",
    "results",
    "experimental results",
    "discussion",
    "conclusion",
)

#: LaTeX scaffolding whose numbers are structural, not empirical.
SKIP_LINE = re.compile(
    r"\\(usepackage|documentclass|geometry|label|ref|cite|includegraphics|"
    r"begin\{equation|end\{equation|section|subsection)"
)

#: Citation identifiers, removed before claims are extracted.
#:
#: "arXiv 2410.21676v4" contains "2410.21", which the number pattern happily
#: reported as an unsourced empirical claim. An arXiv id is a citation, not a
#: measurement, so it never enters the claim set.
CITATION = re.compile(r"arxiv[:\s]*\d{4}\.\d{4,5}(v\d+)?", re.IGNORECASE)

_LATEX_HEADING = re.compile(r"\\section\{([^}]*)\}")
#: Markdown headings, top two levels only. Kept separate from the LaTeX pattern
#: rather than merged into it so the LaTeX path stays byte-identical.
#:
#: Levels 3 and deeper are subsections and do not change which findings-section
#: we are in, matching LaTeX where only ``\section`` ever did. Without the limit,
#: a "### 3. Inspect results" step inside a Usage section reads as a results
#: section and its sample output is scored as an empirical claim.
_MD_HEADING = re.compile(r"^#{1,2}\s+(.+?)\s*#*\s*$")


@dataclass
class Claim:
    value: float
    context: str
    status: str = "unsourced"
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "context": self.context,
            "status": self.status,
            "source": self.source,
        }


def is_claim(token: str) -> bool:
    """Whether a number is an empirical claim rather than scaffolding.

    A year, a layer count, a propagation depth: all numbers, none of them
    results. Requiring either a decimal point or four-plus digits keeps the claim
    set to quantities a run could have measured.
    """
    if "." in token:
        return True
    return len(token) >= 4 and not (1900 <= int(token) <= 2100)


def _heading(line: str) -> str | None:
    """The section this line declares, or ``None`` if it declares none."""
    stripped = line.strip()
    m = _LATEX_HEADING.match(stripped)
    if m:
        return m.group(1).strip().lower()
    m = _MD_HEADING.match(stripped)
    if m:
        return m.group(1).strip().lower()
    return None


def claim_sections(paper_text: str) -> list[str]:
    """Which findings-bearing sections the scanner actually reached.

    Reported alongside any claim count. Zero sections means the scanner could
    not read the document's structure, which is not the same as a document that
    makes no claims, and the two must never render alike.
    """
    found: list[str] = []
    for line in paper_text.splitlines():
        section = _heading(line)
        if section and any(s in section for s in CLAIM_SECTIONS):
            found.append(section)
    return found


def extract_claims(paper_text: str) -> list[Claim]:
    """Numeric claims in the sections where a paper states its findings."""
    claims: list[Claim] = []
    section = "preamble"
    seen: set[tuple[float, str]] = set()
    for line in paper_text.splitlines():
        heading = _heading(line)
        if heading is not None:
            section = heading
            continue
        if SKIP_LINE.search(line):
            continue
        if not any(s in section for s in CLAIM_SECTIONS):
            continue
        line = CITATION.sub(" ", line)
        for token in NUMBER.findall(line):
            if not is_claim(token):
                continue
            value = float(token)
            context = context_of(line, token)
            if (value, context) in seen:
                continue
            seen.add((value, context))
            claims.append(Claim(value=value, context=context))
    return claims


def context_of(line: str, token: str) -> str:
    """The words around a number, for a report a human has to read."""
    i = line.find(token)
    raw = line[max(0, i - 60) : i + len(token) + 30]
    return " ".join(re.sub(r"\\[a-zA-Z]+|[{}$\\]", " ", raw).split())
