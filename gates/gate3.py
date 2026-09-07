"""GATE 3 — REPORT VALIDITY.

> Does every number in the manuscript trace to something that was measured?

Structurally this is Gate 2's sibling, not Gate 1's: it takes an artifact that
already exists, runs nothing, and issues a deterministic verdict. Nothing in
``runner.py`` or ``harness.py`` has an analogue here.

The tiers, each activated by what the caller supplies:

    A  report.no_numeric_literals_in_results   FAIL   always
       report.all_tokens_resolve               FAIL   always
       report.rendered_values_match_registry   FAIL   always
       report.figures_referenced_exist         FAIL   iff the manuscript
                                                      references a figure

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

from .errors import GateError
from .prose import claim_sections, extract_claims
from .registry import citable_values
from .schema import CheckResult, GateReport, Severity, decide

GATE_NAME = "GATE 3 — REPORT VALIDITY"

#: The one thing a writing agent is allowed to emit where a number belongs.
RESULT_TOKEN = re.compile(r"\\result\{([^}]+)\}")

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

    def attempt_dir(self, attempt: int) -> Path:
        return Path(self.artifact_root) / "gate3" / f"attempt_{attempt:02d}"


# --------------------------------------------------------------------------- #
# the renderer
# --------------------------------------------------------------------------- #


def render_result_tokens(
    source: str, values: dict[str, Any]
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
    """
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
    else:
        message = (
            f"no bare numeral in {len(sections)} findings section(s); every "
            f"number came from a result token"
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


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #


def run_gate3(
    source: str,
    registry: dict[str, Any],
    config: Gate3Config,
    attempt: int = 1,
) -> GateReport:
    """Judge one manuscript against the registry Gate 1 wrote.

    ``source`` is the writer's output with its result tokens intact. The gate
    renders it unless the host supplied its own render, then checks that what
    the reader will see is what was measured.
    """
    if not registry.get("citable"):
        raise GateError(
            "Gate 1 did not pass, so no value in this registry is citable and "
            "no manuscript built on it can be verified"
        )
    values = citable_values(registry)

    self_rendered, subs = render_result_tokens(source, values)
    rendered = config.rendered if config.rendered is not None else self_rendered
    origin = "supplied" if config.rendered is not None else "self_rendered"

    checks = [
        _check_no_numeric_literals(source),
        _check_tokens_resolve(source, values),
        _check_rendered_matches(rendered, subs, values, origin),
    ]
    figures = _check_figures_exist(source, config.figure_root)
    if figures is not None:
        checks.append(figures)

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
    (artifact_dir / "gate3_report.json").write_text(
        report.to_json(), encoding="utf-8"
    )
    (artifact_dir / "manuscript.rendered").write_text(rendered, encoding="utf-8")
    return report
