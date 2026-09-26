"""GATE 3 — REPORT VALIDITY.

> Does every number in the manuscript trace to something that was measured?

Structurally this mirrors Gate 1 (D30): a flat list of checks, a
reject-and-retry loop back to the agent that wrote the artifact, and a raise
when the budget is spent. Gate 1 verifies code by running it. Gate 3 verifies a
manuscript's words and sources against the registry Gate 1 wrote, and runs
nothing. The loop is ``report_loop`` in the adapter (D29).

There are no tiers (D7). Families group the checks for discussion only, and
each check runs when its input exists:

    report.no_numeric_literals_in_results   FAIL   always
    report.all_tokens_resolve               FAIL   always
    report.rendered_values_match_registry   FAIL   always
    report.figures_referenced_exist         FAIL   iff the manuscript
                                                   references a figure
    report.limitations_declared             FAIL   iff Gate 2 declared
                                                   limitations
    source.cited_papers_in_registry         FAIL   iff the host says what it
                                                   retrieved and the paper
                                                   cites something
    source.identifiers_resolve              FAIL   iff a resolver is injected
                                                   and the paper cites
                                                   something; INFO if the
                                                   resolver could not run
    style.sections_present                  FAIL   iff the host declares the
                                                   sections it requires
    style.no_orphan_references              FAIL   iff the manuscript
                                                   cross-references or labels
    style.floats_referenced                 WARN   iff the manuscript labels a
                                                   figure or table
    style.claim_sections_bound              FAIL   iff the manuscript has a
                                                   results section
    report.model_unbound_claims             WARN   iff a model is set and
                                                   flags a row; INFO if the
                                                   scan could not run
    report.claim_chains                     INFO   iff a token rendered

``report.claim_chains`` is evidence, not a check: it counts the rendered claims
whose provenance runs unbroken from task to manuscript, and writes each chain to
``claims.json``. A claim whose value is correct but whose run had no task
reference is honest work, so a broken link is reported as a rate and never
blocks.

The model layer is Gate 1's (D31): the model reads what the number scanner
passed and writes the REQUIRED FIXES, and neither can move the verdict.

**What "eliminated by construction" actually means.** The claim in `PLAN.md`
§5.2 is a property of the *pipeline*, not of a scanner: the writer emits
``\\result{key}`` tokens and :func:`render_result_tokens` substitutes registry
values, so a number that was never measured has no token and a token with no
value does not render. ``report.no_numeric_literals_in_results`` is the
enforcement that the pipeline was used, and as a scanner over prose it has a
false-negative rate. The paper must say this in those terms.

The first thing this gate measures is our own archived run: the gated arm's
manuscript types its digits directly, so it fails here. That is the correct
outcome, and it is the number Gate 3 exists to produce.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Callable, Iterable

from . import llm_claims, llm_report
from .errors import GateError
from .llm import (
    DEFAULT_MAX_PROMPT_CHARS,
    DEFAULT_TIMEOUT_S as DEFAULT_MODEL_TIMEOUT_S,
    ModelFn,
    ModelLayer,
)
from .prose import CITATION, claim_sections, extract_claims, sections
from .registry import CHAIN_LINKS, CLAIM_LINK, citable_values, claim_chain
from .schema import CheckResult, GateReport, PaperRecord, Severity, decide

GATE_NAME = "GATE 3 — REPORT VALIDITY"

#: The one thing a writing agent is allowed to emit where a number belongs.
RESULT_TOKEN = re.compile(r"\\result\{([^}]+)\}")

#: Where the writer places Gate 2's declared limitations (D28). Empty braces,
#: like ``\result{key}``, and so TeX does not swallow the space after it.
LIMITATIONS_TOKEN = re.compile(r"\\limitations\{\}")

#: A DOI in the text. D26: the reference host never sees one, so a cited DOI
#: cannot have been retrieved.
_DOI = re.compile(r"\b10\.\d{4,9}/[^\s,;()\[\]{}]+")

#: A retrieved identifier as the host records it, prefix optional.
_ARXIV_ID = re.compile(r"(?:arxiv:?\s*)?(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)

#: What the reader will see, written beside the report on every attempt.
RENDERED_FILENAME = "manuscript.rendered"

#: Each rendered claim's provenance chain, written beside the report when a
#: token rendered.
CLAIMS_FILENAME = "claims.json"

#: Figure references, in both dialects the archived manuscripts actually use.
_FIGURES = (
    re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}"),
    re.compile(r"!\[[^\]]*\]\(([^)]+)\)"),
)


@dataclass(frozen=True)
class Substitution:
    """One token the renderer replaced, and where the result landed."""

    key: str
    start: int
    end: int
    text: str


@dataclass
class Gate3Config:
    """Everything Gate 3 needs. No scaffold types, same as Gates 1 and 2."""

    #: Revisions the writing agent gets. Three, per `PLAN.md` §5 - and giving up
    #: means raising, not proceeding: an unverifiable manuscript is not emitted.
    #: Chosen at setup time; see ``gates/setup.py`` for the cost warning.
    max_attempts: int = 3
    artifact_root: str = "gate_artifacts"
    #: Directory figure paths resolve against. ``None`` resolves relative to the
    #: process's working directory and skips the containment test.
    figure_root: str | None = None
    #: Sections the manuscript must contain, declared by the host at wiring time
    #: (D27). Empty means ``style.sections_present`` emits nothing: which
    #: sections a paper needs is the host's standard, and a default here would be
    #: a preference wearing a check's clothes.
    sections: tuple[str, ...] = ()
    #: Resolves a version-stripped arXiv id to a record, or ``None`` if no such
    #: paper exists. Injected exactly like ``consult_model``, so ``gates/`` never
    #: opens a socket and the suite never depends on someone else's uptime (B2).
    #: ``None`` means ``source.identifiers_resolve`` emits nothing. A resolver
    #: that cannot reach its source raises, and the check reports that it could
    #: not run rather than passing.
    lookup: Callable[[str], PaperRecord | None] | None = None
    #: The host's own render, when the host renders. ``None`` means Gate 3
    #: renders, and says so: a self-rendered comparison is a weaker statement
    #: than an independent one, so its origin travels with it.
    rendered: str | None = None
    #: The model layer's model, as in ``Gate1Config`` (D31). It reads the
    #: findings prose the number scanner passed and drafts the writer's
    #: REQUIRED FIXES, both outside the verdict. Absent, the gate issues the same
    #: verdict, says the scan did not run, and sends the fix template.
    consult_model: ModelFn | None = None
    model_timeout_s: float = DEFAULT_MODEL_TIMEOUT_S
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS

    def attempt_dir(self, attempt: int) -> Path:
        return Path(self.artifact_root) / "gate3" / f"attempt_{attempt:02d}"


# --------------------------------------------------------------------------- #
# the renderer
# --------------------------------------------------------------------------- #


def render_result_tokens(
    source: str, values: dict[str, Any], *, declared: str = ""
) -> tuple[str, list[Substitution]]:
    """Substitute every resolvable ``\\result{key}`` and log what was written.

    Exported so the host's writing loop calls the same function Gate 3 checks.
    Two rules make the check meaningful rather than tautological:

    * **No formatting.** The text written is ``str(value)`` and nothing else. A
      renderer that rounds makes byte-identity false by design, and the archived
      manuscript already carries ``0.9175257731958764`` in full.
    * **An unknown key renders as itself.** Leaving the token verbatim means
      ``report.all_tokens_resolve`` catches it. Substituting an empty string
      would turn a missing measurement into a silently malformed sentence.

    ``declared`` replaces every ``\\limitations{}`` first, so the offsets
    recorded below are offsets in the final text. It goes in a LaTeX
    ``verbatim`` block: a bare ``_`` in a key name stops LaTeX compiling, and a
    ``%`` would comment out the rest of its line, so the limitation would sit in
    the source and never print. LaTeX only, since the reference host writes
    LaTeX; a Markdown host would need a fenced block.
    """
    # limit: a declared text containing \end{verbatim} would close the block
    # early; Gate 2's messages never do, escape it if a host's can.
    if declared and not declared.endswith("\n"):
        declared += "\n"
    block = f"\\begin{{verbatim}}\n{declared}\\end{{verbatim}}" if declared else ""
    # A function, not a string, so re.sub reads no escapes in the block.
    source = LIMITATIONS_TOKEN.sub(lambda _: block, source)
    subs: list[Substitution] = []
    out: list[str] = []
    cursor = 0
    for match in RESULT_TOKEN.finditer(source):
        key = match.group(1).strip()
        entry = values.get(key)
        if entry is None:
            continue
        text = str(entry.get("value") if isinstance(entry, dict) else entry)
        out.append(source[cursor : match.start()])
        start = sum(len(part) for part in out)
        out.append(text)
        subs.append(Substitution(key, start, start + len(text), text))
        cursor = match.end()
    out.append(source[cursor:])
    return "".join(out), subs


# --------------------------------------------------------------------------- #
# tier A
# --------------------------------------------------------------------------- #


def _mask_tokens(source: str) -> str:
    """Blank the tokens so the scanner sees only what the model typed."""
    return RESULT_TOKEN.sub("RESULT", source)


def _check_no_numeric_literals(source: str) -> CheckResult:
    """No bare numeral appears where the paper states its findings.

    Coverage decides the verdict's meaning. A manuscript whose findings sections
    the scanner cannot locate has not been checked, and reporting that as a pass
    would be the same defect this gate exists to catch. It is reported as INFO
    and degraded instead - absent, never green.
    """
    masked = _mask_tokens(source)
    sections = claim_sections(masked)
    if not sections:
        return CheckResult(
            id="report.no_numeric_literals_in_results",
            passed=True,
            severity=Severity.INFO,
            message=(
                "no findings section could be located, so no numeric literal "
                "was looked for"
            ),
            evidence={"sections_scanned": [], "degraded": True},
        )

    literals = [
        {"value": c.value, "context": c.context} for c in extract_claims(masked)
    ]
    if literals:
        message = (
            f"{len(literals)} bare numeral(s) in a results context, e.g. "
            f"{literals[0]['value']!r} in \"{literals[0]['context']}\""
        )
    elif RESULT_TOKEN.search(source):
        message = (
            f"no bare numeral in {len(sections)} findings section(s); every "
            f"number came from a result token"
        )
    else:
        # Saying "every number came from a token" here would describe numbers
        # that do not exist. style.claim_sections_bound judges the absence.
        message = (
            f"no bare numeral in {len(sections)} findings section(s), and no "
            f"result token either: the manuscript states no number at all"
        )

    return CheckResult(
        id="report.no_numeric_literals_in_results",
        passed=not literals,
        severity=Severity.FAIL,
        message=message,
        evidence={
            "literals": literals,
            "sections_scanned": sections,
            "discrepancies": [
                f"{row['value']!r} is typed into the manuscript rather than "
                f"cited from a measurement: \"{row['context']}\""
                for row in literals
            ],
        },
    )


def _check_tokens_resolve(source: str, values: dict[str, Any]) -> CheckResult:
    """Every result token names a key the registry actually holds."""
    tokens = sorted({m.group(1).strip() for m in RESULT_TOKEN.finditer(source)})
    missing = sorted(set(tokens) - set(values))
    return CheckResult(
        id="report.all_tokens_resolve",
        passed=not missing,
        severity=Severity.FAIL,
        message=(
            f"{len(missing)} result token(s) name no recorded value: "
            f"{', '.join(missing)}"
            if missing
            else f"{len(tokens)} result token(s) resolve to a recorded value"
        ),
        # Same evidence shape as Gate 1's expected_keys_present, so report.py's
        # existing renderer displays it with no new code.
        evidence={
            "missing": missing,
            "recorded": sorted(values),
            "tokens": tokens,
        },
    )


def _check_rendered_matches(
    rendered: str,
    subs: list[Substitution],
    values: dict[str, Any],
    origin: str,
) -> CheckResult:
    """Every substituted value is byte-identical to the registry value.

    ``origin`` says whether the rendered text came from the host or from this
    gate. Self-rendered is the weaker claim and is labelled as such, the same
    move ``Band.origin`` makes in Gate 2.
    """
    mismatches: list[dict[str, Any]] = []
    for sub in subs:
        entry = values.get(sub.key)
        expected = str(entry.get("value") if isinstance(entry, dict) else entry)
        actual = rendered[sub.start : sub.end]
        if actual != expected:
            mismatches.append(
                {"key": sub.key, "expected": expected, "actual": actual}
            )

    if mismatches:
        first = mismatches[0]
        message = (
            f"{len(mismatches)} rendered value(s) differ from the registry, "
            f"e.g. {first['key']}: registry holds {first['expected']!r}, the "
            f"manuscript reads {first['actual']!r}"
        )
    else:
        message = (
            f"{len(subs)} substituted value(s) byte-identical to the registry "
            f"({origin})"
        )

    return CheckResult(
        id="report.rendered_values_match_registry",
        passed=not mismatches,
        severity=Severity.FAIL,
        message=message,
        evidence={
            "mismatches": mismatches,
            "substituted": len(subs),
            "origin": origin,
            "discrepancies": [
                f"{m['key']} reads {m['actual']!r} in the manuscript but "
                f"{m['expected']!r} in the registry"
                for m in mismatches
            ],
        },
    )


def _check_figures_exist(source: str, figure_root: str | None) -> CheckResult | None:
    """Every referenced figure is on disk, and inside the run's own artifacts.

    Returns ``None`` when the manuscript references no figure. A check with
    nothing to check emits nothing rather than a green row.
    """
    referenced: list[str] = []
    for pattern in _FIGURES:
        referenced.extend(m.group(1).strip() for m in pattern.finditer(source))
    if not referenced:
        return None

    root = Path(figure_root).resolve() if figure_root else None
    missing: list[dict[str, str]] = []
    for target in sorted(set(referenced)):
        path = (root / target) if root else Path(target)
        if not path.exists():
            missing.append({"target": target, "reason": "not on disk"})
        elif root and not path.resolve().is_relative_to(root):
            # A figure reached by escaping the run's artifact directory was not
            # produced by the gated run, whatever it shows.
            missing.append({"target": target, "reason": "outside the run's artifacts"})

    return CheckResult(
        id="report.figures_referenced_exist",
        passed=not missing,
        severity=Severity.FAIL,
        message=(
            f"{len(missing)} referenced figure(s) unavailable, e.g. "
            f"{missing[0]['target']} ({missing[0]['reason']})"
            if missing
            else f"{len(set(referenced))} referenced figure(s) present"
        ),
        evidence={
            "missing": missing,
            "referenced": sorted(set(referenced)),
            "figure_root": str(root) if root else None,
            "discrepancies": [
                f"{m['target']}: {m['reason']}" for m in missing
            ],
        },
    )


def _check_limitations_declared(
    source: str, rendered: str, declared: str, origin: str
) -> CheckResult | None:
    """Every line Gate 2 declared appears, word for word, in what the reader sees.

    D28: rendered, not authored. The writer places ``\\limitations{}`` and the
    renderer inserts the text, because no deterministic check can tell a
    faithful paraphrase from a softened one. Lines are compared stripped, so a
    host renderer that re-indents still passes and one that drops or edits a
    line does not. Returns ``None`` when there is nothing to declare.
    """
    lines = [line.strip() for line in declared.splitlines() if line.strip()]
    if not lines:
        return None
    missing = [line for line in lines if line not in rendered]
    token_found = bool(LIMITATIONS_TOKEN.search(source))
    if missing:
        message = f"{len(missing)} of {len(lines)} declared limitation line(s) are not in the manuscript"
        if not token_found:
            message += ", which has no \\limitations{} token"
    else:
        message = f"{len(lines)} declared limitation line(s) stated word for word ({origin})"
    return CheckResult(
        id="report.limitations_declared",
        passed=not missing,
        severity=Severity.FAIL,
        message=message,
        evidence={
            "missing": missing,
            "token_found": token_found,
            "origin": origin,
            "discrepancies": [
                f"Gate 2 declared a limitation the manuscript does not state: {line}"
                for line in missing
            ],
        },
    )


def arxiv_versions(ids: Iterable[str]) -> dict[str, set[str]]:
    """Retrieved arXiv ids as ``{id: {versions}}``; ``""`` is an unversioned one.

    Anything that is not an arXiv id is dropped: D26 makes the arXiv id the
    only identifier a citation is compared by.
    """
    versions: dict[str, set[str]] = {}
    for raw in ids:
        match = _ARXIV_ID.fullmatch(str(raw).strip())
        if match:
            versions.setdefault(match.group(1), set()).add(match.group(2) or "")
    return versions


def _check_cited_papers_in_registry(
    source: str, retrieved: Iterable[str] | None
) -> CheckResult | None:
    """Every cited paper is one the run retrieved (D21, D25, D26).

    A citation nobody's search or review returned is MLR-Bench's incorrect
    citation, and this is a membership test, so it needs no network. Ids are
    compared without their version: citing v4 of a paper the run read as v2
    passes, and the difference is kept as evidence. A cited DOI fails, because
    the reference host never sees one.

    Returns ``None`` when the host gave no retrieval record, or when the
    manuscript cites nothing.
    """
    if retrieved is None:
        return None
    cited = sorted({m.group(1) + (m.group(2) or "") for m in CITATION.finditer(source)})
    dois = sorted({m.group(0).rstrip(".") for m in _DOI.finditer(source)})
    if not cited and not dois:
        return None

    versions = arxiv_versions(retrieved)
    not_retrieved: list[str] = []
    mismatches: list[dict[str, Any]] = []
    for citation in cited:
        base, _, version = citation.partition("v")
        version = f"v{version}" if version else ""
        if base not in versions:
            not_retrieved.append(citation)
        elif version and version not in versions[base] and versions[base] - {""}:
            mismatches.append({
                "cited": citation,
                "retrieved": sorted(base + v for v in versions[base] if v),
            })
    not_retrieved += dois

    if not_retrieved:
        message = (
            f"{len(not_retrieved)} of {len(cited) + len(dois)} cited paper(s) were "
            f"never retrieved during the run, e.g. {not_retrieved[0]}"
        )
    else:
        message = f"{len(cited)} cited paper(s), all retrieved during the run"
        if mismatches:
            message += f"; {len(mismatches)} cite a different version than the one read"
    return CheckResult(
        id="source.cited_papers_in_registry",
        passed=not not_retrieved,
        severity=Severity.FAIL,
        message=message,
        evidence={
            "cited": cited + dois,
            "not_retrieved": not_retrieved,
            "version_mismatches": mismatches,
            "retrieved": sorted(base + v for base, vs in versions.items() for v in vs),
            "discrepancies": [
                f"{paper} is cited but no search or review during the run returned it"
                for paper in not_retrieved
            ],
        },
    )


#: LaTeX's abstract environment. ``prose._heading`` deliberately does not see it
#: (D40): presence is a different question from claim scanning, and teaching the
#: scanner to read inside it would restate the published Gate 1 number. The
#: unscanned abstract is G3-M4's to report, not this check's to close.
_ABSTRACT_ENV = re.compile(r"\\begin\{abstract\}")


def _check_sections_present(source: str, declared: tuple[str, ...]) -> CheckResult | None:
    """Every section the host declared at wiring time is in the manuscript.

    D27: the list is the host's, because which sections a paper needs is its
    standard and not a fact ``gates/`` knows. A declared name is present when
    some heading contains it, so a host declaring ``"results"`` accepts
    "Experimental Results". The check asks whether the section is there, not
    whether the writer named it our way.

    Returns ``None`` when the host declared nothing.
    """
    if not declared:
        return None
    headings = [heading for heading, _ in sections(source)]
    if _ABSTRACT_ENV.search(source):
        headings.append("abstract")
    missing = [name for name in declared if not any(name in h for h in headings)]
    return CheckResult(
        id="style.sections_present",
        passed=not missing,
        severity=Severity.FAIL,
        message=(
            f"{len(missing)} of {len(declared)} declared section(s) are not in "
            f"the manuscript: {', '.join(missing)}"
            if missing
            else f"all {len(declared)} declared section(s) present"
        ),
        evidence={
            "missing": missing,
            "declared": list(declared),
            "headings": [h for h in headings if h != "preamble"],
            "discrepancies": [
                f"the host requires a {name!r} section and the manuscript has none"
                for name in missing
            ],
        },
    )


#: Cross-reference commands LaTeX resolves against a ``\label``, and the labels
#: themselves. ``cleveref``'s ``\cref{a,b}`` holds a comma list, so targets are
#: split: reading it as one target named "a,b" would invent an orphan and miss
#: the real one.
_REF = re.compile(r"\\(?:auto|c|C|eq|page)?ref\{([^}]+)\}")
_LABEL = re.compile(r"\\label\{([^}]+)\}")


def _check_no_orphan_references(source: str) -> CheckResult | None:
    """Every cross-reference resolves to a label the manuscript defines.

    D33, and FAIL because a dangling ``\\ref`` renders as "??" in the PDF: a
    reader sees it, so it is a defect rather than a preference. An unreferenced
    label is not judged here. A float nobody points at is
    ``style.floats_referenced``'s, at WARN, and a label on a section nobody
    points at is nobody's.

    Returns ``None`` when the manuscript neither references nor labels anything.
    """
    labels = {m.group(1).strip() for m in _LABEL.finditer(source)}
    referenced = {
        target.strip()
        for m in _REF.finditer(source)
        for target in m.group(1).split(",")
        if target.strip()
    }
    if not referenced and not labels:
        return None
    orphans = sorted(referenced - labels)
    if orphans:
        message = (
            f"{len(orphans)} cross-reference(s) name no label in the "
            f"manuscript: {', '.join(orphans)}"
        )
    elif not referenced:
        message = (
            f"{len(labels)} label(s) defined and no cross-reference to resolve"
        )
    else:
        message = f"{len(referenced)} cross-reference(s) resolve to a label"
    return CheckResult(
        id="style.no_orphan_references",
        passed=not orphans,
        severity=Severity.FAIL,
        message=message,
        evidence={
            "orphans": orphans,
            "referenced": sorted(referenced),
            "labels": sorted(labels),
            "discrepancies": [
                f"the manuscript references {name!r} and defines no such label, "
                f"so it renders as \"??\""
                for name in orphans
            ],
        },
    )


#: A float and the labels inside it. Non-greedy and DOTALL because the reference
#: host writes multi-line environments, so the label rarely sits on the ``\begin``
#: line. ``figure*`` and ``table*`` count: the star changes the column span, not
#: whether a reader needs to be sent to it.
_FLOAT = re.compile(
    r"\\begin\{(figure|table)\*?\}(.*?)\\end\{\1\*?\}", re.DOTALL
)


def _check_floats_referenced(source: str) -> CheckResult | None:
    """Every labelled figure and table is pointed at from the text.

    D33, and WARN rather than FAIL: a float the prose never mentions is a
    drafting slip, not an unverifiable claim, and costing the writer a turn for
    it could cost the paper on the last attempt. A float with no label at all is
    not counted, because there is no way to reference one and demanding a label
    is a preference D27 refuses without a host saying so.

    Returns ``None`` when the manuscript labels no float.
    """
    labelled: list[dict[str, str]] = []
    for match in _FLOAT.finditer(source):
        kind, body = match.group(1), match.group(2)
        for label in _LABEL.finditer(body):
            labelled.append({"label": label.group(1).strip(), "kind": kind})
    if not labelled:
        return None
    referenced = {
        target.strip()
        for m in _REF.finditer(source)
        for target in m.group(1).split(",")
    }
    unreferenced = [row for row in labelled if row["label"] not in referenced]
    return CheckResult(
        id="style.floats_referenced",
        passed=not unreferenced,
        severity=Severity.WARN,
        message=(
            f"{len(unreferenced)} of {len(labelled)} labelled float(s) are never "
            f"referenced in the text, e.g. {unreferenced[0]['label']}"
            if unreferenced
            else f"all {len(labelled)} labelled float(s) referenced in the text"
        ),
        evidence={
            "unreferenced": unreferenced,
            "labelled": labelled,
            "discrepancies": [
                f"the {row['kind']} labelled {row['label']!r} is never "
                f"referenced, so a reader is never sent to it"
                for row in unreferenced
            ],
        },
    )


def _check_identifiers_resolve(
    source: str, lookup: Callable[[str], PaperRecord | None] | None
) -> CheckResult | None:
    """Every cited arXiv id resolves to a paper that exists.

    The sibling of ``source.cited_papers_in_registry``, asking the other half of
    the question. That one catches a paper this run never retrieved; this catches
    an identifier no such paper was ever issued. A citation can fail either.

    Ids are looked up version-stripped (D26): whether v4 specifically exists is
    what the run read, which the registry check already judges. DOIs are never
    sent, because arXiv does not issue them and a cited DOI already fails the
    registry check.

    Returns ``None`` when the host injected no resolver, or the paper cites
    nothing.

    **An outage cannot launder a fabrication.** Every identifier is asked about,
    and a resolver failure on one does not discard what the others established.
    Only when nothing was found unresolved *and* something could not be asked
    does this degrade to an INFO row: an outage is not a defect in the
    manuscript, so it must not block it, and it must not read as a pass either.
    Returning early on the first failure would let a paper citing one fabricated
    id and one unreachable id through, which is duty 1.
    """
    if lookup is None:
        return None
    cited = sorted({m.group(1) for m in CITATION.finditer(source)})
    if not cited:
        return None

    resolved: list[dict[str, Any]] = []
    unresolved: list[str] = []
    unchecked: list[str] = []
    reasons: list[str] = []
    for identifier in cited:
        try:
            record = lookup(identifier)
        except Exception as exc:  # noqa: BLE001 - any resolver fault is "cannot ask"
            unchecked.append(identifier)
            reasons.append(f"{type(exc).__name__}: {exc}"[:120])
            continue
        if record is None:
            unresolved.append(identifier)
        else:
            resolved.append({
                "identifier": record.identifier,
                "title": record.title,
                "year": record.year,
                "locator": record.locator,
            })

    evidence: dict[str, Any] = {
        "unresolved": unresolved,
        "resolved": resolved,
        "unchecked": unchecked,
        "discrepancies": [
            f"{name} is cited but no paper with that identifier exists"
            for name in unresolved
        ],
    }
    if not unresolved and unchecked:
        evidence["degraded"] = True
        evidence["reason"] = "; ".join(dict.fromkeys(reasons))
        return CheckResult(
            id="source.identifiers_resolve",
            passed=True,
            severity=Severity.INFO,
            message=(
                f"{len(unchecked)} of {len(cited)} cited identifier(s) could not "
                f"be resolved: the resolver failed"
            ),
            evidence=evidence,
        )

    message = (
        f"{len(unresolved)} of {len(cited)} cited identifier(s) name no "
        f"existing paper: {', '.join(unresolved)}"
        if unresolved
        else f"all {len(cited)} cited identifier(s) resolve to a real paper"
    )
    if unchecked:
        # Said on the failing row too, so the writer is not left thinking the
        # named ids were the only problem.
        message += f"; {len(unchecked)} could not be checked"
        evidence["reason"] = "; ".join(dict.fromkeys(reasons))
    return CheckResult(
        id="source.identifiers_resolve",
        passed=not unresolved,
        severity=Severity.FAIL,
        message=message,
        evidence=evidence,
    )


def _check_claim_sections_bound(
    source: str, values: dict[str, Any]
) -> CheckResult | None:
    """Every results section cites at least one measured value.

    The degenerate evasion: a writer rejected for typed numbers stops writing
    numbers, and every binding check above passes a paper that reports nothing
    measured. Only sections whose heading contains "results" are held to it. A
    discussion with no number in it is honest writing, and failing it would
    cost the writer a turn, or on the last turn the paper.

    Returns ``None`` when there is no results section. A missing Results heading
    is ``style.sections_present``'s to catch, when the host declares its
    sections (D27). Tokens count whether or not they resolve, since
    ``report.all_tokens_resolve`` already fails an unknown key.
    """
    counted = [
        {"section": heading, "tokens": len(RESULT_TOKEN.findall(body))}
        for heading, body in sections(source)
        if "results" in heading
    ]
    if not counted:
        return None
    unbound = [row["section"] for row in counted if not row["tokens"]]
    return CheckResult(
        id="style.claim_sections_bound",
        passed=not unbound,
        severity=Severity.FAIL,
        message=(
            f"{len(unbound)} results section(s) cite no measured value: "
            f"{', '.join(unbound)}"
            if unbound
            else f"{len(counted)} results section(s) each cite a measured value"
        ),
        evidence={
            "sections": counted,
            "unbound": unbound,
            # The keys the writer can cite, so the fix is actionable.
            "recorded": sorted(values),
            "discrepancies": [
                f"the {name!r} section states its results without citing a "
                f"single measured value"
                for name in unbound
            ],
        },
    )


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #


def claim_chains(
    rendered: str, subs: list[Substitution], registry: dict[str, Any]
) -> list[dict[str, Any]]:
    """One record per rendered token: where it landed, and its full chain.

    The claim link resolves when the reader sees the registry value at the
    token's position, the same byte comparison
    ``report.rendered_values_match_registry`` makes.
    """
    values = registry.get("values") or {}
    claims = []
    for sub in subs:
        entry = values.get(sub.key) or {}
        expected = str(entry.get("value"))
        seen = rendered[sub.start : sub.end]
        chain = claim_chain(
            registry,
            sub.key,
            ref=f"{RENDERED_FILENAME}:{sub.start}-{sub.end}",
            resolved=seen == expected,
            why=None if seen == expected else f"the manuscript reads {seen!r} here",
        )
        claims.append(
            {
                "key": sub.key,
                "start": sub.start,
                "end": sub.end,
                "trace_id": entry.get("trace_id"),
                "chain": chain,
                "chain_complete": all(link["resolved"] for link in chain),
            }
        )
    return claims


def _check_claim_chains(claims: list[dict[str, Any]], path: Path) -> CheckResult | None:
    """The traced-claim rate. INFO: a broken link is reported, never blocking."""
    if not claims:
        return None
    complete = sum(1 for c in claims if c["chain_complete"])
    missing: dict[str, int] = {}
    for claim in claims:
        for link in claim["chain"]:
            if not link["resolved"]:
                missing[link["link"]] = missing.get(link["link"], 0) + 1
    order = (*CHAIN_LINKS, CLAIM_LINK)
    message = (
        f"{complete} of {len(claims)} rendered claim(s) trace unbroken through "
        f"{', '.join(order)}"
    )
    if missing:
        message += "; unresolved: " + ", ".join(
            f"{name} ({missing[name]})" for name in order if name in missing
        )
    return CheckResult(
        id="report.claim_chains",
        passed=complete == len(claims),
        severity=Severity.INFO,
        message=message,
        evidence={
            "claims": len(claims),
            "complete_chains": complete,
            "rate": complete / len(claims),
            "missing_links": {name: missing[name] for name in order if name in missing},
            "path": str(path),
        },
    )


def run_gate3(
    source: str,
    registry: dict[str, Any],
    config: Gate3Config,
    attempt: int = 1,
    *,
    declared: str = "",
    retrieved: Iterable[str] | None = None,
) -> GateReport:
    """Judge one manuscript against the registry Gate 1 wrote.

    ``source`` is the writer's output with its result tokens intact. The gate
    renders it unless the host supplied its own render, then checks that what
    the reader will see is what was measured. ``declared`` is Gate 2's
    ``ReviewOutcome.declared``, the limitations the manuscript must state.
    ``retrieved`` is every paper id the host's searches and review returned;
    ``None`` means the host does not say, and citations go unchecked.
    """
    retrieved = None if retrieved is None else set(retrieved)
    if not registry.get("citable"):
        raise GateError(
            "Gate 1 did not pass, so no value in this registry is citable and "
            "no manuscript built on it can be verified"
        )
    values = citable_values(registry)

    self_rendered, subs = render_result_tokens(source, values, declared=declared)
    rendered = config.rendered if config.rendered is not None else self_rendered
    origin = "supplied" if config.rendered is not None else "self_rendered"

    checks = [
        _check_no_numeric_literals(source),
        _check_tokens_resolve(source, values),
        _check_rendered_matches(rendered, subs, values, origin),
    ]
    model = ModelLayer(
        config.consult_model,
        timeout_s=config.model_timeout_s,
        max_prompt_chars=config.max_prompt_chars,
    )
    for optional in (
        _check_figures_exist(source, config.figure_root),
        _check_limitations_declared(source, rendered, declared, origin),
        _check_cited_papers_in_registry(source, retrieved),
        _check_identifiers_resolve(source, config.lookup),
        _check_sections_present(source, config.sections),
        _check_no_orphan_references(source),
        # WARN by construction, so decide() below is blind to it.
        _check_floats_referenced(source),
        _check_claim_sections_bound(source, values),
        # WARN or INFO by construction, so decide() below is blind to it.
        llm_claims.build_check(llm_claims.scan_claims(model, _mask_tokens(source))),
    ):
        if optional is not None:
            checks.append(optional)

    artifact_dir = config.attempt_dir(attempt)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    claims = claim_chains(rendered, subs, registry)
    chains = _check_claim_chains(claims, artifact_dir / CLAIMS_FILENAME)
    if chains is not None:
        # INFO by construction, so decide() below is blind to it.
        checks.append(chains)
        (artifact_dir / CLAIMS_FILENAME).write_text(
            json.dumps(claims, indent=2), encoding="utf-8"
        )

    report = GateReport(
        gate=GATE_NAME,
        verdict=decide(checks),
        attempt=attempt,
        max_attempts=config.max_attempts,
        checks=checks,
        artifact_dir=str(artifact_dir),
    )
    # The verdict is fixed above. Only now is the model asked to write anything
    # the writer will read, as in Gate 1.
    if not report.passed and model.available:
        keys = sorted(values)
        papers = sorted(retrieved or ())
        facts = "CITABLE KEYS: " + ", ".join(keys)
        if retrieved is not None:
            facts += "\nRETRIEVED PAPERS: " + ", ".join(papers)
        # A fix may name a retrieved paper, or a cited one it says to remove.
        known = set(arxiv_versions(papers)) | {m.group(1) for m in CITATION.finditer(source)}
        llm_report.attach_fixes(
            report,
            llm_report.generate_fixes(
                model,
                report,
                source,
                system=llm_report.REPORT_SYSTEM,
                facts=facts,
                grounding=lambda text: llm_report.check_manuscript_grounding(
                    text, report, source, keys, papers=known
                ),
            ),
            subject="this manuscript and its registry",
        )
    if model.available:
        report.model = model.budget.to_dict()
    (artifact_dir / "gate3_report.json").write_text(
        report.to_json(), encoding="utf-8"
    )
    (artifact_dir / RENDERED_FILENAME).write_text(rendered, encoding="utf-8")
    return report
