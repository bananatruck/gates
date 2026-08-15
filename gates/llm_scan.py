"""Model-assisted log scanning — the recall half of the LLM layer.

The deterministic pattern set finds 18 of the 34 signals in the labelled corpus
at perfect precision. The 16 it misses are not pattern bugs; they are shapes no
regex was going to reach:

    Error loading graphs-datasets/cora: Dataset ... cannot be accessed
    Skipping 3 of 10 folds that raised during fitting; mean is over the remaining 7
    Note: test set was used for early stopping because no validation split was provided
    epoch 012  loss nan  train_acc 0.1429

Each says, in plain language, that the numbers around it mean something other
than what they appear to mean. Reading that is what a model is for.

Two constraints shape everything here:

**Precision is the floor, not recall.** A false positive costs the ML engineer a
rewrite for nothing. The corpus deliberately includes TensorFlow's "Unable to
register cuFFT factory" and oneDNN's round-off banner — E-level, CUDA-adjacent,
mentioning "errors", and present on every healthy run. A scanner that raises
recall by flagging those has made the gate worse.

**Findings are grounded.** A returned line number that does not exist, or a
quoted line that does not match the capture, is discarded rather than reported.
The model reports *which* lines are signals; it does not get to invent them.

Severity is WARN, by construction — see ``llm.py``. Nothing here can move a
verdict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from . import log_digest, retrieval
from .llm import ModelLayer, model_warning
from .log_checks import LogFinding
from .schema import CheckResult, Severity

CHECK_ID = "logs.model_error_signals"

#: Lines sent to the model per stream. Bounds cost, and is reported so the
#: recall claim is bounded by what was actually examined.
DEFAULT_MAX_LINES = 400

#: Findings kept. Beyond a handful the report stops being small, and the ML
#: engineer stops reading it.
MAX_FINDINGS = 6

#: Retrieved exemplars per class. Zero disables few-shot entirely, which is
#: the arm the bench compares against.
DEFAULT_FEW_SHOT = 3

_SYSTEM = (
    "You audit the logs of a machine-learning experiment that already finished. "
    "Your job is to find lines reporting trouble the run CONTINUED PAST — "
    "anything that changes what the run's reported numbers actually mean.\n\n"
    "Report a line only if a careful reviewer would need to know about it before "
    "believing the results. Examples of what counts: a dataset that failed to "
    "load and was substituted, a metric printed as nan or inf, replicates "
    "silently dropped from an average, a test split used for model selection, a "
    "fallback to different weights or a different device when a timing claim is "
    "being made, an exception that was caught and swallowed.\n\n"
    "Do NOT report routine framework noise. Library registration messages, "
    "deprecation and future warnings, progress bars, ordinary training logs, and "
    "metric names that merely contain the word 'error' are not signals. A false "
    "positive costs an engineer a wasted rewrite, so when a line is merely "
    "noisy, leave it out.\n\n"
    "Respond with a JSON array and nothing else. Each element: "
    '{"line": <the line number as given>, "why": "<at most 20 words on what it '
    'means for the reported numbers>"}. '
    "If nothing qualifies, respond with []."
)


@dataclass(frozen=True)
class ScanOutcome:
    """What the model contributed, and how much of the log it actually saw."""

    findings: list[LogFinding]
    lines_examined: int
    lines_skipped: int
    ok: bool
    error: str | None = None
    raw: str = ""
    #: Distinct shapes sent, after collapsing. The difference between this and
    #: ``lines_examined`` is what repetition was costing.
    shapes_sent: int = 0
    #: Exemplars retrieved into the prompt, for the bench to attribute by.
    exemplars_used: int = 0

    @property
    def compression(self) -> float:
        if not self.lines_examined:
            return 0.0
        return 1.0 - (self.shapes_sent / self.lines_examined)


def scan_with_model(
    layer: ModelLayer,
    stdout: str,
    stderr: str,
    *,
    already_flagged: list[LogFinding] | None = None,
    max_lines: int = DEFAULT_MAX_LINES,
    few_shot: int = DEFAULT_FEW_SHOT,
) -> ScanOutcome:
    """Ask the model about the lines the pattern set did not already flag.

    Excluding the flagged lines is not only a cost measure: re-reporting a
    finding the deterministic tier already made would inflate the model's
    apparent contribution, which is the number this whole exercise is trying to
    measure honestly.
    """
    flagged = {(f.stream, f.lineno) for f in (already_flagged or [])}
    candidates: list[tuple[str, int, str]] = []
    skipped = 0

    for stream, text in (("stdout", stdout), ("stderr", stderr)):
        kept = 0
        for lineno, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not line or (stream, lineno) in flagged:
                continue
            if kept >= max_lines:
                skipped += 1
                continue
            candidates.append((stream, lineno, line))
            kept += 1

    if not candidates:
        return ScanOutcome([], 0, skipped, ok=True)

    # Collapse before sending. A 200-epoch loop is one shape, and the model
    # reads the two lines that matter instead of hunting for them.
    shapes = log_digest.digest(candidates)
    # Findings resolve against the real first line, not the rendered row: the
    # "[x200 identical]" annotation is prompt scaffolding and has no business in
    # the evidence the engineer reads.
    rows = [(s.stream, s.lineno, s.line) for s in shapes]

    # One number per row, deliberately. An earlier format showed both the row
    # index and the file line -- "3<TAB>[stdout:202]<TAB>Skipping 3 of 10 folds"
    # -- and qwen3:8b reported 202, the file line, which is out of range as an
    # index and was silently discarded by the grounding check. The finding was
    # correct and the recall was lost to prompt ambiguity. gemini-3.5-flash
    # happened to read it the intended way, which is exactly why a weaker model
    # is worth testing against. The file line is not information the model needs
    # to judge a line, and this side resolves it anyway.
    numbered = "\n".join(
        f"{i}\t[{s.stream}]\t{s.render()}"
        for i, s in enumerate(shapes, start=1)
    )
    # Retrieval-augmented few-shot: the exemplars nearest this particular log,
    # balanced across signal and noise. The bank is disjoint from the evaluation
    # corpus, so this teaches the boundary rather than leaking the answers.
    system = _SYSTEM
    exemplars: list = []
    if few_shot:
        exemplars = retrieval.select(numbered, k_each=few_shot)
        block = retrieval.render(exemplars)
        if block:
            system = f"{_SYSTEM}\n\n{block}"

    call = layer.ask(
        "Experiment log, one row per distinct line. Rows identical in shape "
        "have been collapsed into one and marked [xN].\n\n"
        "The first column is the ROW NUMBER. Report row numbers from that "
        f"column and nothing else; valid values are 1 to {len(shapes)}.\n\n"
        f"{numbered}",
        system,
    )
    if not call.ok:
        return ScanOutcome(
            [], len(candidates), skipped, ok=False, error=call.error,
            shapes_sent=len(shapes),
        )

    findings = _parse(call.text, [(s, ln, t) for s, ln, t in rows])
    return ScanOutcome(
        findings=findings,
        lines_examined=len(candidates),
        lines_skipped=skipped,
        ok=True,
        raw=call.text,
        shapes_sent=len(shapes),
        exemplars_used=len(exemplars),
    )


def _parse(text: str, candidates: list[tuple[str, int, str]]) -> list[LogFinding]:
    """Turn the completion into findings, discarding anything ungrounded.

    Tolerant of a model that wraps its JSON in prose or a code fence, and
    strict about indices: an index outside the range it was given is a
    hallucination and is dropped rather than repaired.
    """
    payload = _extract_json_array(text)
    if payload is None:
        return []

    findings: list[LogFinding] = []
    seen: set[int] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("line"))
        except (TypeError, ValueError):
            continue
        if not 1 <= index <= len(candidates) or index in seen:
            continue
        seen.add(index)
        stream, lineno, line = candidates[index - 1]
        why = str(item.get("why") or "").strip()
        findings.append(
            LogFinding(
                signal="model_flagged",
                note=why or "the model flagged this line as material to the results",
                stream=stream,
                lineno=lineno,
                line=line,
            )
        )
        if len(findings) >= MAX_FINDINGS:
            break
    return findings


def _extract_json_array(text: str) -> list | None:
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, list) else None


def build_check(outcome: ScanOutcome) -> CheckResult | None:
    """The check the pipeline appends, or ``None`` when there is nothing to say.

    A model that found nothing is not a passing check worth a line in the
    report; it is silence. A model that could not be reached *is* worth
    recording, because it bounds what the report is allowed to claim.
    """
    if not outcome.ok:
        return CheckResult(
            id=CHECK_ID,
            passed=True,
            severity=Severity.INFO,
            message=(
                "the log scan could not run, so error signals outside the "
                "deterministic pattern set were not looked for"
            ),
            evidence={"error": outcome.error, "degraded": True},
        )
    if not outcome.findings:
        return None
    return model_warning(
        CHECK_ID,
        (
            f"{len(outcome.findings)} line(s) outside the deterministic pattern "
            f"set report trouble the run continued past"
        ),
        {
            "findings": [
                {
                    "signal": f.signal,
                    "note": f.note,
                    "stream": f.stream,
                    "lineno": f.lineno,
                    "line": f.line,
                }
                for f in outcome.findings
            ],
            "lines_examined": outcome.lines_examined,
            "lines_skipped": outcome.lines_skipped,
        },
    )
