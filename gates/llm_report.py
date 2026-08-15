"""The feedback report's REQUIRED FIXES, written by the model — and grounded.

This is why Gate 1 has an LLM layer. ``PLAN.md`` §2 requires every gate to emit
"a small natural-language feedback report for the agent it loops back to," and
§3.4 shows what that has to mean::

    Bind `hidden_dim` before use, or pass it into `GCN.__init__`.
    It is read at line 337 but never assigned in any enclosing scope.

``report._FIXES`` holds one canned sentence per check id. It cannot produce that,
because the fix depends on the code: where the name is read, what scope it is in,
what the enclosing class takes. A template that says "bind every name listed
above" is not wrong, it is just not worth reading — and a feedback report the
engineer skims is a rewrite spent for nothing.

**Grounding is the whole risk, and it is checked mechanically.** A generated fix
that names a variable the code does not contain sends the engineer chasing
something that does not exist — the exact failure Gate 1 exists to prevent,
reintroduced at its exit. So every identifier the model puts in backticks and
every line number it cites must appear in the facts it was given. Ungrounded
output is rejected whole and the template renders instead; a partly-invented fix
is not repaired into a real one.

Ordinary prose is not policed. The rule constrains the specific and checkable —
names and line numbers — and leaves English alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm import ModelLayer
from .schema import GateReport

#: Identifiers the model may always use: they are not claims about this code.
_ALWAYS_ALLOWED = frozenset(
    {
        "record_result",
        "record_metadata",
        "results.json",
        "seed",
        "stdout",
        "stderr",
        "None",
        "True",
        "False",
        "int",
        "float",
        "str",
        "list",
        "dict",
        "nan",
        "inf",
        "NaN",
        "self",
        "__init__",
        "__main__",
    }
)

_BACKTICKED = re.compile(r"`([^`\n]{1,80})`")
_LINE_REFERENCE = re.compile(r"\bline\s+(\d+)\b", re.IGNORECASE)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_STRING_LITERAL = re.compile(r"'[^']*'|\"[^\"]*\"")

#: A backticked span that is a call to one of these is a *template* for code the
#: engineer is being told to write, not a reference to code that already exists.
#: `record_result("exp1.acc", acc)` proposes a key and a variable; it does not
#: claim either is already there, so grounding it against the current source
#: would make the one fix `results.contract_present` needs impossible to write.
_TEMPLATE_APIS = ("record_result", "record_metadata")

_SYSTEM = (
    "You write the REQUIRED FIXES section of a code-validity gate's feedback "
    "report. The reader is an automated ML engineer agent that will rewrite the "
    "experiment and resubmit it.\n\n"
    "Write one numbered instruction per problem, in the order given. Each must "
    "say what to change and where. Be concrete: name the variable, the function "
    "and the line. Prefer the smallest correct fix.\n\n"
    "Rules you must not break:\n"
    "- Use ONLY names, line numbers and facts that appear in the report below. "
    "Never invent a variable, function, file or line number.\n"
    "- Put code identifiers in backticks.\n"
    "- Do not restate the diagnosis; the reader already has it. Give the fix.\n"
    "- Do not apologise, praise, score, or speculate about intent.\n"
    "- No preamble and no closing remarks. Output the numbered list only.\n"
    "- At most 4 instructions, at most 40 words each."
)


@dataclass
class FixOutcome:
    """The generated section, and whether it survived grounding."""

    text: str = ""
    ok: bool = False
    #: Tokens the model used that are not in the report's facts. Non-empty means
    #: the text was rejected.
    ungrounded: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def usable(self) -> bool:
        return self.ok and bool(self.text.strip()) and not self.ungrounded


def generate_fixes(
    layer: ModelLayer, report: GateReport, source: str = ""
) -> FixOutcome:
    """Ask the model for REQUIRED FIXES, and refuse anything ungrounded."""
    failures = report.failed_checks()
    if not failures:
        return FixOutcome(ok=True, text="")
    if not layer.available:
        return FixOutcome(ok=False, error="no model was supplied")

    call = layer.ask(_render_facts(report), _SYSTEM)
    if not call.ok:
        return FixOutcome(ok=False, error=call.error)

    text = _strip_preamble(call.text)
    ungrounded = check_grounding(text, report, source)
    return FixOutcome(text=text, ok=True, ungrounded=ungrounded)


# --------------------------------------------------------------------------- #
# grounding
# --------------------------------------------------------------------------- #


def build_vocabulary(report: GateReport, source: str = "") -> tuple[set[str], set[int]]:
    """Every name and line number the model is permitted to cite.

    Drawn from the checks' own evidence and from the submitted source, because
    a fix that names a real function in the code under repair is legitimate
    while one that names a function from some other program is not.
    """
    names: set[str] = set(_ALWAYS_ALLOWED)
    linenos: set[int] = set()

    if source:
        for match in _IDENTIFIER.finditer(source):
            names.add(match.group(0))
        linenos.update(range(1, source.count("\n") + 2))

    for check in report.checks:
        names.add(check.id)
        names.update(_IDENTIFIER.findall(check.id))
        _harvest(check.evidence, names, linenos)

    execution = report.execution
    if execution is not None:
        names.update(execution.metrics)
        for key in execution.metrics:
            names.update(_IDENTIFIER.findall(key))
        names.update(str(k) for k in execution.metadata)
        if execution.exception is not None:
            names.add(execution.exception.type)
            names.update(_IDENTIFIER.findall(execution.exception.message or ""))
            if execution.exception.lineno:
                linenos.add(execution.exception.lineno)
    return names, linenos


def _harvest(node: object, names: set[str], linenos: set[int]) -> None:
    """Walk arbitrary evidence, collecting identifiers and line numbers."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("lineno", "line") and isinstance(value, int):
                linenos.add(value)
            _harvest(value, names, linenos)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _harvest(item, names, linenos)
    elif isinstance(node, str):
        names.update(_IDENTIFIER.findall(node))
    elif isinstance(node, int) and not isinstance(node, bool):
        linenos.add(node)


def check_grounding(text: str, report: GateReport, source: str = "") -> list[str]:
    """Tokens the model used that the report does not support.

    Only backticked identifiers and explicit line references are checked. A
    sentence of ordinary English asserts nothing verifiable and is left alone.
    """
    names, linenos = build_vocabulary(report, source)
    offenders: list[str] = []

    for match in _BACKTICKED.finditer(text):
        span = match.group(1).strip()
        if span.startswith(_TEMPLATE_APIS):
            continue
        # String literals are proposed values, not references to existing names,
        # so they are stripped before identifiers are extracted. What remains is
        # checked identifier by identifier rather than as a whole span, so that
        # `GCN.__init__` and `self.lin1(x)` are not rejected over punctuation.
        for identifier in _IDENTIFIER.findall(_STRING_LITERAL.sub(" ", span)):
            if identifier not in names:
                offenders.append(identifier)

    for match in _LINE_REFERENCE.finditer(text):
        lineno = int(match.group(1))
        if linenos and lineno not in linenos:
            offenders.append(f"line {lineno}")

    seen: set[str] = set()
    return [x for x in offenders if not (x in seen or seen.add(x))]


# --------------------------------------------------------------------------- #
# prompt rendering
# --------------------------------------------------------------------------- #


def _render_facts(report: GateReport) -> str:
    """The facts the model may draw on. Deliberately only the report's own."""
    lines = [f"VERDICT: {report.verdict.value}", "", "FAILED CHECKS", ""]
    for check in report.failed_checks():
        lines.append(f"[{check.id}] {check.message}")
        lines.extend(_evidence_lines(check.evidence))
        lines.append("")

    warnings = report.warnings()
    if warnings:
        lines.append("WARNINGS (do not write fixes for these; context only)")
        for check in warnings:
            lines.append(f"[{check.id}] {check.message}")
        lines.append("")

    execution = report.execution
    if execution is not None and execution.exception is not None:
        exc = execution.exception
        lines += [
            "UNCAUGHT EXCEPTION",
            f"  {exc.type}: {exc.message}",
            f"  at line {exc.lineno} in {exc.function or '<module>'}",
        ]
        if exc.source_line:
            lines.append(f"  {exc.lineno} | {exc.source_line.strip()}")
        lines.append("")

    if execution is not None and execution.metrics:
        lines.append("RECORDED METRICS: " + ", ".join(sorted(execution.metrics)))
    elif execution is not None:
        lines.append("RECORDED METRICS: none")
    return "\n".join(lines)


def _evidence_lines(evidence: dict, limit: int = 6) -> list[str]:
    out: list[str] = []
    for key, value in evidence.items():
        if isinstance(value, list):
            for item in value[:limit]:
                out.append(f"  - {_flatten(item)}")
        elif value not in (None, "", {}):
            out.append(f"  {key}: {_flatten(value)}")
    return out


def _flatten(item: object) -> str:
    if isinstance(item, dict):
        return "  ".join(f"{k}={v}" for k, v in item.items() if v not in (None, ""))
    return str(item)


def _strip_preamble(text: str) -> str:
    """Keep the numbered list. Models add a sentence of throat-clearing even
    when told not to, and the report has no room for it."""
    lines = text.strip().splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if re.match(r"\s*\d+[.)]\s", ln)), None
    )
    if start is None:
        return text.strip()
    kept: list[str] = []
    for line in lines[start:]:
        if line.strip().startswith("```"):
            break
        kept.append(line.rstrip())
    return "\n".join(kept).strip()
