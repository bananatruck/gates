"""Tier C — the semantic half of Gate 2, and the half that is not provable.

Two questions no deterministic check reaches:

* ``coherence.method_match`` — does the implemented method match what the cited
  source describes? Same normalization, same splits, same objective.
* ``coherence.claim_supported`` — is each claim the plan says these results
  establish actually entailed by the measured values?

Both are WARN by construction, and the reason is a published number rather than
caution for its own sake. BadScientist puts LLM detection of this class of
defect at approximately chance. A tier built on that cannot be allowed to decide
anything, and a paper that reports it must report it as a **rate with an
interval**, never as a guarantee — `PLAN.md` §4.4.

So the arrangement is Gate 1's, unchanged: findings are constructed through
:func:`gates.llm.model_warning`, which does not take a severity. ``Severity.FAIL``
does not appear in this module and a test asserts that it never does.

**Grounding.** The model reports *which* of the things it was shown are
mismatched; it does not get to invent them. A finding naming a source id, a
claim index or a registry key that was not in the prompt is discarded rather
than reported. Gate 1 learned this the expensive way: a generated fix that
invents a variable name costs the engineer a rewrite, which is the failure the
layer exists to prevent, reintroduced at its exit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .llm import ModelLayer, model_warning
from .schema import CheckResult, Severity

METHOD_CHECK_ID = "coherence.method_match"
CLAIM_CHECK_ID = "coherence.claim_supported"

#: Findings kept per check. Past a handful the feedback report stops being small
#: and the engineer stops reading it — the same ceiling ``llm_scan`` uses.
MAX_FINDINGS = 6

#: The implemented method is sent as text. Bounded here as well as in
#: ``ModelLayer`` so the ceiling is visible at the call site.
DEFAULT_MAX_METHOD_CHARS = 12_000

_METHOD_SYSTEM = (
    "You are comparing a machine-learning experiment's implementation against "
    "the papers it cites. Report only places where the implementation does "
    "something DIFFERENT from what the cited source describes, in a way that "
    "would change the numbers: a different normalization, a different train/"
    "test split, a different objective or loss, a different evaluation protocol, "
    "a different preprocessing step.\n\n"
    "Do NOT report cosmetic differences — variable names, library choice, code "
    "structure, ordering — or anything you are merely unsure about. A false "
    "report sends the engineer to rewrite working code, so when in doubt, say "
    "nothing.\n\n"
    "You may only refer to the source ids listed in the prompt. Do not mention "
    "any other paper.\n\n"
    "Respond with a JSON array and nothing else. Each element: "
    '{"source_id": "<one of the listed ids>", "aspect": "<what differs, 3-6 '
    'words>", "why": "<at most 25 words on how it would change the numbers>"}. '
    "If the implementation matches every source, respond with []."
)

_CLAIM_SYSTEM = (
    "You are checking whether an experiment's measured values support the "
    "claims its plan says they establish. You are given the claims, each with "
    "an index, and the full set of measured values.\n\n"
    "Report only claims the values do NOT support: a claim about a trend "
    "measured at a single point, a comparison where only one side was measured, "
    "a magnitude the numbers contradict, a causal statement the measurements "
    "cannot separate. A claim that the values do support must not be reported.\n\n"
    "Judge only against the values given. Do not use outside knowledge of what "
    "these methods usually score.\n\n"
    "Respond with a JSON array and nothing else. Each element: "
    '{"claim": <the claim index as given>, "why": "<at most 25 words on what '
    'the values do not establish>"}. '
    "If every claim is supported, respond with []."
)


@dataclass(frozen=True)
class Finding:
    """One thing the model flagged, after grounding."""

    ref: str
    note: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"ref": self.ref, "note": self.note, "detail": self.detail}


@dataclass
class Outcome:
    """What one semantic pass produced. ``ok=False`` degrades, never blocks."""

    ok: bool
    findings: list[Finding] = field(default_factory=list)
    error: str | None = None
    #: How many the model returned that grounding threw away. Reported, because
    #: a model that mostly invents references is worth knowing about.
    discarded: int = 0
    examined: int = 0


# --------------------------------------------------------------------------- #
# coherence.method_match
# --------------------------------------------------------------------------- #


def scan_method_match(
    model: ModelLayer,
    method_source: str,
    sources: list[dict[str, Any]],
    *,
    max_method_chars: int = DEFAULT_MAX_METHOD_CHARS,
) -> Outcome:
    """Ask whether the implementation departs from what its sources describe."""
    if not model.available:
        return Outcome(ok=False, error="no model was supplied")
    if not method_source.strip() or not sources:
        return Outcome(
            ok=False,
            error="no implementation text or no cited source to compare it against",
        )

    known = {str(s.get("source_id")) for s in sources if s.get("source_id")}
    listed = "\n".join(
        f"- {s.get('source_id')}: {s.get('describes') or s.get('setting') or ''}".rstrip()
        for s in sources
    )
    prompt = (
        f"CITED SOURCES\n{listed}\n\n"
        f"IMPLEMENTATION\n{method_source[:max_method_chars]}\n"
    )

    parsed, error = _ask_json(model, prompt, _METHOD_SYSTEM)
    if error:
        return Outcome(ok=False, error=error)

    findings: list[Finding] = []
    discarded = 0
    for item in parsed:
        source_id = str(item.get("source_id", "")).strip()
        # Grounding: a source the prompt never listed is an invention, and a
        # feedback report that cites one sends the engineer to a paper that was
        # never retrieved.
        if source_id not in known:
            discarded += 1
            continue
        aspect = str(item.get("aspect", "")).strip()
        why = str(item.get("why", "")).strip()
        if not aspect and not why:
            discarded += 1
            continue
        findings.append(Finding(ref=source_id, note=aspect or why, detail=why))

    return Outcome(
        ok=True,
        findings=findings[:MAX_FINDINGS],
        discarded=discarded,
        examined=len(sources),
    )


def build_method_check(outcome: Outcome) -> CheckResult | None:
    """The check to append, or ``None`` when there is nothing to say."""
    if not outcome.ok:
        return _unavailable(
            METHOD_CHECK_ID,
            "the method-match pass could not run, so a departure from the cited "
            "sources' protocol was not looked for",
            outcome,
        )
    if not outcome.findings:
        return None
    return model_warning(
        METHOD_CHECK_ID,
        f"{len(outcome.findings)} place(s) where the implementation may depart "
        f"from what a cited source describes",
        _evidence(outcome),
    )


# --------------------------------------------------------------------------- #
# coherence.claim_supported
# --------------------------------------------------------------------------- #


def scan_claims_supported(
    model: ModelLayer,
    claims: list[str],
    values: dict[str, Any],
) -> Outcome:
    """Ask whether the measured values entail the claims the plan declared."""
    if not model.available:
        return Outcome(ok=False, error="no model was supplied")
    if not claims:
        return Outcome(ok=False, error="the plan declared no claims to check")

    listed_claims = "\n".join(f"[{i}] {c}" for i, c in enumerate(claims))
    listed_values = "\n".join(
        f"- {k} = {v.get('value')!r}"
        + (f" ({v.get('unit')})" if v.get("unit") else "")
        for k, v in values.items()
    )
    prompt = (
        f"CLAIMS\n{listed_claims}\n\n"
        f"MEASURED VALUES\n{listed_values or '(none recorded)'}\n"
    )

    parsed, error = _ask_json(model, prompt, _CLAIM_SYSTEM)
    if error:
        return Outcome(ok=False, error=error)

    findings: list[Finding] = []
    discarded = 0
    for item in parsed:
        index = item.get("claim")
        # Grounding: an index outside the list is an invention. Accepting it
        # would put a claim into the report that the plan never made.
        if not isinstance(index, int) or not 0 <= index < len(claims):
            discarded += 1
            continue
        why = str(item.get("why", "")).strip()
        if not why:
            discarded += 1
            continue
        findings.append(Finding(ref=str(index), note=claims[index], detail=why))

    return Outcome(
        ok=True,
        findings=findings[:MAX_FINDINGS],
        discarded=discarded,
        examined=len(claims),
    )


def build_claim_check(outcome: Outcome) -> CheckResult | None:
    if not outcome.ok:
        return _unavailable(
            CLAIM_CHECK_ID,
            "the claim-support pass could not run, so whether the values entail "
            "the plan's claims was not looked for",
            outcome,
        )
    if not outcome.findings:
        return None
    return model_warning(
        CLAIM_CHECK_ID,
        f"{len(outcome.findings)} plan claim(s) the measured values may not "
        f"establish",
        _evidence(outcome),
    )


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _ask_json(
    model: ModelLayer, prompt: str, system: str
) -> tuple[list[dict[str, Any]], str | None]:
    """One call, parsed. Any failure is an error string, never an exception."""
    call = model.ask(prompt, system)
    if not call.ok:
        return [], call.error or "the model call failed"
    try:
        parsed = json.loads(_json_slice(call.text))
    except (ValueError, TypeError):
        return [], "the model did not return parseable JSON"
    if not isinstance(parsed, list):
        return [], "the model did not return a JSON array"
    return [item for item in parsed if isinstance(item, dict)], None


def _json_slice(text: str) -> str:
    """The outermost JSON array in a completion.

    Small models wrap the array in prose or a fenced block however firmly the
    prompt says not to. Slicing to the brackets is cheaper than a retry and
    cannot fabricate a finding: if there is no array, parsing fails and the pass
    degrades.
    """
    start = text.find("[")
    end = text.rfind("]")
    return text[start : end + 1] if 0 <= start < end else text


def _evidence(outcome: Outcome) -> dict[str, Any]:
    return {
        "findings": [f.to_dict() for f in outcome.findings],
        "examined": outcome.examined,
        "discarded_ungrounded": outcome.discarded,
        "discrepancies": [
            f"{f.ref}: {f.detail or f.note}" for f in outcome.findings
        ],
    }


def _unavailable(check_id: str, message: str, outcome: Outcome) -> CheckResult:
    """A pass that could not run is INFO, not a green check.

    Recording it is what stops a report claiming coverage it did not have. It
    passes because an absent model is not the agent's fault, and it is INFO
    rather than WARN because nothing was found — the finding is that nothing was
    looked for.
    """
    return CheckResult(
        id=check_id,
        passed=True,
        severity=Severity.INFO,
        message=message,
        evidence={"error": outcome.error, "degraded": True},
    )
