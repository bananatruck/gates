"""Check a generated paper's numbers against what its run actually recorded.

The archived Agent Laboratory paper reports 81.60% test accuracy, a 13.61x
speedup and a 39.20% ablation collapse. Its saved experiment code contains none
of those numbers and calls no recording API at all. The paper reads well; every
headline figure in it is unsourced. That is the failure this project exists to
catch, and catching it by hand once is not a measurement -- so it is automated
here and both papers are put through the same procedure.

The procedure, for each numeric claim in the paper's abstract, contributions and
results:

1. **Is it in the registry?** A recorded result matches within tolerance -- the
   claim is *sourced*.
2. **Is it in the experiment code's output?** A printed value the writer copied
   is *printed* -- correct, perhaps, but bound to nothing.
3. **Neither?** *Unsourced*. It came from the model.

A number being unsourced is not proof it is wrong. It is proof that nothing in
the pipeline could have told the difference, which is the point being made.

Deliberately excluded from the claim set: citation years, equation indices,
hyperparameters the paper itself specifies, and small integers, none of which are
empirical claims. Getting this wrong in the permissive direction would inflate
the unsourced count and flatter the argument, so the filter errs the other way.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from gates.prose import (
    CITATION,
    CLAIM_SECTIONS,
    NUMBER,
    SKIP_LINE,
    Claim,
    context_of,
    extract_claims,
    is_claim,
)

# The prose scanner moved to ``gates/prose.py``: Gate 3 needs it and
# ``pyproject.toml`` packages ``gates*`` only, so ``gates`` cannot import
# ``rig``. Re-exported under their original private names so this module's
# callers and tests are unaffected. The move also fixed a Markdown blind spot -
# see that module's docstring.
_NUMBER = NUMBER
_CLAIM_SECTIONS = CLAIM_SECTIONS
_SKIP_LINE = SKIP_LINE
_CITATION = CITATION
_is_claim = is_claim
_context = context_of


@dataclass
class PaperAudit:
    paper: str
    claims: list[Claim] = field(default_factory=list)
    registry_keys: list[str] = field(default_factory=list)
    record_result_calls: int = 0
    note: str = ""

    def count(self, status: str) -> int:
        return sum(1 for c in self.claims if c.status == status)

    @property
    def sourced_rate(self) -> float | None:
        return self.count("sourced") / len(self.claims) if self.claims else None

    @property
    def traceable_rate(self) -> float | None:
        """Sourced or checkably derived from something sourced."""
        if not self.claims:
            return None
        n = self.count("sourced") + self.count("derived")
        return n / len(self.claims)

    def to_dict(self) -> dict:
        return {
            "paper": self.paper,
            "n_claims": len(self.claims),
            "sourced": self.count("sourced"),
            "derived": self.count("derived"),
            "printed": self.count("printed"),
            "unsourced": self.count("unsourced"),
            "sourced_rate": self.sourced_rate,
            "traceable_rate": self.traceable_rate,
            "registry_keys": self.registry_keys,
            "record_result_calls": self.record_result_calls,
            "note": self.note,
            "claims": [c.to_dict() for c in self.claims],
        }


def _close(a: float, b: float) -> bool:
    """A paper rounds; 81.60 and 0.816 are the same measurement."""
    for scale in (1.0, 100.0, 0.01):
        x = b * scale
        if abs(a - x) <= max(0.011 * abs(a), 1e-6):
            return True
    return False


def derived_from(value: float, registry: dict[str, float]) -> str | None:
    """Whether `value` is simple arithmetic over recorded results.

    A paper that records accuracy and then writes "the error drops from 0.11 to
    0.03" has not invented anything: 1 - 0.89 and 1 - 0.97 are the recorded
    numbers, restated. Scoring those as unsourced puts a legitimate derivation
    in the same bucket as a fabricated one, and the whole argument here rests on
    that distinction being real.

    So the arithmetic is *checked*, never assumed -- only a relationship that
    actually holds earns the label, and anything else stays unsourced. The set
    is deliberately small: complement, difference, ratio, and percentage. A
    wider search would eventually "explain" any number at all, which would make
    the category meaningless in the direction that flatters the gated arm.
    """
    items = list(registry.items())
    for key, v in items:
        if _close(value, 1.0 - v):
            return f"1 - {key}"
    for a, va in items:
        for b, vb in items:
            if a == b:
                continue
            if _close(value, va - vb):
                return f"{a} - {b}"
            if vb and _close(value, va / vb):
                return f"{a} / {b}"
            if vb and _close(value, (1.0 - va) / (1.0 - vb)) and vb != 1.0:
                return f"(1-{a}) / (1-{b})"
    return None


def load_registry(path: str | Path) -> dict[str, float]:
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get("values") or data.get("metrics") or {}
    out = {}
    for key, entry in values.items():
        v = entry.get("value") if isinstance(entry, dict) else entry
        if isinstance(v, (int, float)):
            out[key] = float(v)
    return out


#: Agent Laboratory writes these into its workflow log. Their presence means the
#: capture is a transcript of everything the agents said, not a program's output.
_WORKFLOW_MARKERS = (
    "Beginning phase:", "Beginning subtask:", "Subtask '",
    "Current experiment cost", "@@@ NOVEL", "~~~~~~~~~~~",
)


def _contaminated(capture: str) -> bool:
    """Whether `capture` is a workflow transcript rather than a program's output.

    This matters more than it looks. A whole-workflow log contains every agent's
    prose, and the interpretation agent restates the paper's headline numbers in
    it. Scored against such a log, all 65 of the archived paper's claims come
    back "printed" -- which reads as though the run had produced them, when what
    actually happened is that a model wrote the same number twice.

    Checking for the paper verbatim is not enough: the log states the figures in
    prose while the paper states them in LaTeX, so a text probe misses and the
    contamination passes silently. What is reliable is recognising the container.
    """
    return sum(m in capture for m in _WORKFLOW_MARKERS) >= 2


def audit(paper_path: str | Path, *, registry_path: str | Path | None = None,
          code_output: str = "", code_text: str = "") -> PaperAudit:
    paper_path = Path(paper_path)
    result = PaperAudit(paper=str(paper_path))
    if not paper_path.exists():
        result.note = "paper not found"
        return result

    text = paper_path.read_text(errors="replace")
    result.claims = extract_claims(text)
    registry = load_registry(registry_path) if registry_path else {}
    result.registry_keys = sorted(registry)
    result.record_result_calls = len(
        re.findall(r"\brecord_result\s*\(", code_text)
    )

    if code_output and _contaminated(code_output):
        result.note = ("capture is a workflow transcript, not an execution "
                       "capture — the agents' own prose restates the paper's "
                       "figures, so 'printed' is not measurable and is not claimed")
        code_output = ""
    printed = {float(t) for t in _NUMBER.findall(code_output) if _is_claim(t)}
    for claim in result.claims:
        hit = next((k for k, v in registry.items() if _close(claim.value, v)), None)
        if hit:
            claim.status, claim.source = "sourced", f"registry:{hit}"
            continue
        if any(_close(claim.value, p) for p in printed):
            claim.status, claim.source = "printed", "stdout, untraceable"
            continue
        how = derived_from(claim.value, registry)
        if how:
            claim.status, claim.source = "derived", how
    return result


def render(a: PaperAudit) -> str:
    lines = [
        f"paper: {a.paper}",
        f"  numeric claims in findings sections : {len(a.claims)}",
        f"  sourced to a recorded result        : {a.count('sourced')}",
        f"  derived from recorded results       : {a.count('derived')}",
        f"  printed but untraceable             : {a.count('printed')}",
        f"  unsourced (no origin at all)        : {a.count('unsourced')}",
        f"  record_result calls in the code     : {a.record_result_calls}",
        f"  registry keys                       : {', '.join(a.registry_keys) or '(none)'}",
    ]
    if a.note:
        lines.append(f"  note: {a.note}")
    unsourced = [c for c in a.claims if c.status == "unsourced"][:12]
    if unsourced:
        lines.append("  unsourced claims:")
        lines += [f"    {c.value:<12g} ...{c.context}..." for c in unsourced]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    paper = sys.argv[1]
    registry = sys.argv[2] if len(sys.argv) > 2 else None
    code = Path(sys.argv[3]).read_text(errors="replace") if len(sys.argv) > 3 else ""
    out = audit(paper, registry_path=registry, code_text=code, code_output=code)
    print(render(out))
