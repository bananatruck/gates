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
    style.claim_sections_bound              FAIL   iff the manuscript has a
                                                   results section
    report.model_unbound_claims             WARN   iff a model is set and
                                                   flags a row; INFO if the
                                                   scan could not run

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

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import llm_claims, llm_report
from .errors import GateError
from .llm import (
    DEFAULT_MAX_PROMPT_CHARS,
    DEFAULT_TIMEOUT_S as DEFAULT_MODEL_TIMEOUT_S,
    ModelFn,
    ModelLayer,
)
from .prose import claim_sections, extract_claims, sections
from .registry import citable_values
from .schema import CheckResult, GateReport, Severity, decide

GATE_NAME = "GATE 3 — REPORT VALIDITY"

#: The one thing a writing agent is allowed to emit where a number belongs.
RESULT_TOKEN = re.compile(r"\\result\{([^}]+)\}")

#: Where the writer places Gate 2's declared limitations (D28). Empty braces,
#: like ``\result{key}``, and so TeX does not swallow the space after it.
LIMITATIONS_TOKEN = re.compile(r"\\limitations\{\}")

#: What the reader will see, written beside the report on every attempt.
RENDERED_FILENAME = "manuscript.rendered"

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
    # ponytail: a declared text containing \end{verbatim} would close the block
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


def run_gate3(
    source: str,
    registry: dict[str, Any],
    config: Gate3Config,
    attempt: int = 1,
    *,
    declared: str = "",
) -> GateReport:
    """Judge one manuscript against the registry Gate 1 wrote.

    ``source`` is the writer's output with its result tokens intact. The gate
    renders it unless the host supplied its own render, then checks that what
    the reader will see is what was measured. ``declared`` is Gate 2's
    ``ReviewOutcome.declared``, the limitations the manuscript must state.
    """
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
        _check_claim_sections_bound(source, values),
        # WARN or INFO by construction, so decide() below is blind to it.
        llm_claims.build_check(llm_claims.scan_claims(model, _mask_tokens(source))),
    ):
        if optional is not None:
            checks.append(optional)

    artifact_dir = config.attempt_dir(attempt)
    artifact_dir.mkdir(parents=True, exist_ok=True)

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
        llm_report.attach_fixes(
            report,
            llm_report.generate_fixes(
                model,
                report,
                source,
                system=llm_report.REPORT_SYSTEM,
                facts="CITABLE KEYS: " + ", ".join(keys),
                grounding=lambda text: llm_report.check_manuscript_grounding(
                    text, report, source, keys
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
