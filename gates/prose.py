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

LaTeX behaviour was frozen until D76, because the published Gate 1
traceability number came from this scanner reading ``.tex`` (D38). D76 masks
references and citations per token instead of skipping their lines, and
re-measures both numbers it restates: G3-M4 from 34 to 46 of 49, and Gate 1
traceability on the 08-15 campaign's gated paper from 28 of 29 claims to 34 of
37. The signed package keeps its vintage.

D104 reads LaTeX's abstract environment as the abstract, and gives each repeat
of a value on one line its own context, after both level 3 manuscripts of
waves 2-3 turned out to write ``\\begin{abstract}``: a number typed there
passed Gate 3 unread. Re-measured: G3-M4 47 of 49, and Gate 1 traceability
on the 08-15 gated paper 35 of 38 (ungated 15 of 22, archived 0 of 69).
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

#: LaTeX scaffolding whose numbers are structural, not empirical. A line
#: carrying one is not prose, so every number on it is skipped.
SKIP_LINE = re.compile(
    r"\\(usepackage|documentclass|geometry|includegraphics|"
    r"begin\{equation|end\{equation|section|subsection)"
)

#: A cross-reference, citation or label inside a sentence: ``\ref{tab:2}``,
#: ``\eqref{eq:1}``, ``\citep[p.~3]{smith2020}``, ``\label{fig:3}``. Its digits
#: are an identifier, so the command is masked and the rest of the line is
#: still read (D76). Until D76 any such command skipped the whole line, which
#: hid 12 of G3-M4's 15 missed results.
REFERENCE = re.compile(
    r"\\(?:[A-Za-z]*ref|[A-Za-z]*cite[A-Za-z]*|label)\*?(?:\[[^\]]*\])*\{[^}]*\}"
)


def scannable(line: str) -> str | None:
    """The line as the claim scanner reads it, or ``None`` when it is scaffolding.

    References and citations are blanked token by token, and arXiv ids with
    them; what remains is read for numbers.
    """
    if SKIP_LINE.search(line):
        return None
    return CITATION.sub(" ", REFERENCE.sub(" ", line))

#: Citation identifiers, removed before claims are extracted.
#:
#: "arXiv 2410.21676v4" contains "2410.21", which the number pattern happily
#: reported as an unsourced empirical claim. An arXiv id is a citation, not a
#: measurement, so it never enters the claim set.
#:
#: Group 1 is the id and group 2 its version, for Gate 3's citation check.
#: Old-style ids (``hep-th/9901001``) are not matched.
CITATION = re.compile(r"arxiv[:\s]*(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)

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


#: LaTeX's abstract environment. ``_heading`` does not see it, so the line
#: walk below does: it is the abstract, a findings section (D104).
_ABSTRACT_BEGIN = re.compile(r"\\begin\{abstract\}")
_ABSTRACT_END = re.compile(r"\\end\{abstract\}")


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
        elif _ABSTRACT_BEGIN.search(line):
            found.append("abstract")
    return found


def findings_lines(paper_text: str) -> list[tuple[int, str, str]]:
    """``(line number, section, text)`` for every line of a findings section.

    A ``\\begin{abstract}`` block is the abstract (D104). Until D104 the
    walk followed headings only, so an abstract written as an environment was
    never read, and a number typed there passed Gate 3. Text on the
    environment's own begin and end lines is kept; the markers are not.
    """
    out: list[tuple[int, str, str]] = []
    section = "preamble"
    in_abstract = False
    for number, line in enumerate(paper_text.splitlines(), 1):
        heading = _heading(line)
        if heading is not None:
            section = heading
            continue
        text = line
        begin = _ABSTRACT_BEGIN.search(text)
        if begin:
            in_abstract = True
            text = text[begin.end():]
        end = _ABSTRACT_END.search(text) if in_abstract else None
        if end:
            text = text[: end.start()]
        current = "abstract" if in_abstract else section
        if end:
            in_abstract = False
        if any(s in current for s in CLAIM_SECTIONS) and text.strip():
            out.append((number, current, text))
    return out


def line_claim_tokens(line: str) -> list[tuple[str, str]]:
    """``(token, context)`` for each claim on one line, as the scanner reads it.

    Each occurrence gets the words around its own position, so a value stated
    twice on one line is two claims (D104); ``line.find`` used to give every
    repeat the first one's context, and the repeat was deduplicated away.
    """
    text = scannable(line)
    if text is None:
        return []
    out: list[tuple[str, str]] = []
    for match in NUMBER.finditer(text):
        token = match.group(1)
        if not is_claim(token):
            continue
        pair = (token, _context_at(text, match.start(), match.end()))
        if pair not in out:
            out.append(pair)
    return out


def flags_claim(line: str) -> bool:
    """Whether :func:`extract_claims` takes a number from this findings line."""
    text = scannable(line)
    if text is None:
        return False
    return any(is_claim(token) for token in NUMBER.findall(text))


def sections(paper_text: str) -> list[tuple[str, str]]:
    """Each top-level section as ``(heading, body)``, in order.

    Text before the first heading is ``"preamble"``. Subsections stay in their
    section's body, for the same reason :func:`_heading` ignores them.
    """
    out: list[tuple[str, list[str]]] = [("preamble", [])]
    for line in paper_text.splitlines():
        heading = _heading(line)
        if heading is None:
            out[-1][1].append(line)
        else:
            out.append((heading, []))
    return [(heading, "\n".join(body)) for heading, body in out]


def extract_claims(paper_text: str) -> list[Claim]:
    """Numeric claims in the sections where a paper states its findings."""
    claims: list[Claim] = []
    seen: set[tuple[str, str]] = set()
    for _, _, text in findings_lines(paper_text):
        for token, context in line_claim_tokens(text):
            if (token, context) in seen:
                continue
            seen.add((token, context))
            claims.append(Claim(value=float(token), context=context))
    return claims


def context_of(line: str, token: str) -> str:
    """The words around a number's first occurrence, for a human to read."""
    i = line.find(token)
    return _context_at(line, i, i + len(token))


def _context_at(line: str, start: int, end: int) -> str:
    raw = line[max(0, start - 60) : end + 30]
    return " ".join(re.sub(r"\\[a-zA-Z]+|[{}$\\]", " ", raw).split())
