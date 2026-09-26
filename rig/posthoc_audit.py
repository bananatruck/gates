"""Audit papers other systems already released, with no run behind them (D61).

Figure 7 in ``paper/PLAN.md``. Gate 3's ``source.identifiers_resolve`` needs
neither a run nor a registry, so it can ask of any released paper whether each
arXiv identifier it cites names a paper that exists. The published numbers
disagree: MLR-Bench found incorrect citations in 30% of AI Scientist v2's
papers, ScientistOne's audit found none in 159 references. One check over all
of them is how the paper settles it.

Layout, one folder per system, text only::

    <root>/<system>/<paper>.tex | .txt | .md

A PDF must be converted first (``pdftotext``); ``gates/`` and ``rig/`` stay
stdlib-only, and a PDF parser would be a dependency.

**What this measures, exactly.** Only citations written as arXiv identifiers
are checkable. A paper that cites none is counted as not checkable, never as
clean: absent, not green. A paper whose lookups all failed is degraded and left
out of the rate, and the count of those is reported beside it.

    python -m rig.posthoc_audit papers/ --out .cache/posthoc_audit.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from gates.gate3 import audit_identifiers
from gates.schema import PaperRecord
from rig.stats import wilson

Lookup = Callable[[str], PaperRecord | None]

PAPER_SUFFIXES = (".tex", ".txt", ".md")


@dataclass
class PaperResult:
    system: str
    paper: str
    #: ``unresolved``: cites an id no paper has. ``resolved``: every cited id
    #: exists. ``degraded``: nothing unresolved, but a lookup failed.
    #: ``not_checkable``: no arXiv identifier to ask about.
    outcome: str
    cited: int = 0
    unresolved: list[str] = field(default_factory=list)
    unchecked: list[str] = field(default_factory=list)


def audit_paper(path: Path, system: str, lookup: Lookup) -> PaperResult:
    check = audit_identifiers(path.read_text(encoding="utf-8", errors="replace"), lookup)
    if check is None:
        return PaperResult(system, path.name, "not_checkable")
    evidence = check.evidence
    unresolved = list(evidence["unresolved"])
    unchecked = list(evidence["unchecked"])
    cited = len(unresolved) + len(unchecked) + len(evidence["resolved"])
    if unresolved:
        outcome = "unresolved"
    elif evidence.get("degraded"):
        outcome = "degraded"
    else:
        outcome = "resolved"
    return PaperResult(system, path.name, outcome, cited, unresolved, unchecked)


def audit_tree(root: Path, lookup: Lookup) -> list[PaperResult]:
    results = []
    for system_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for paper in sorted(system_dir.iterdir()):
            if paper.suffix in PAPER_SUFFIXES:
                results.append(audit_paper(paper, system_dir.name, lookup))
    return results


def summarise(results: list[PaperResult]) -> dict[str, dict]:
    """Per system: the share of checked papers citing an id that does not exist."""
    out: dict[str, dict] = {}
    for system in sorted({r.system for r in results}):
        rows = [r for r in results if r.system == system]
        count = {k: sum(r.outcome == k for r in rows) for k in
                 ("unresolved", "resolved", "degraded", "not_checkable")}
        checked = count["unresolved"] + count["resolved"]
        lo, hi = wilson(count["unresolved"], checked)
        out[system] = {
            "papers": len(rows),
            **count,
            "checked": checked,
            "rate": count["unresolved"] / checked if checked else None,
            "ci95": [round(lo, 4), round(hi, 4)],
        }
    return out


def render(summary: dict[str, dict]) -> str:
    lines = [
        "RELEASED PAPERS CITING AN arXiv ID THAT DOES NOT EXIST",
        "",
        f"  {'SYSTEM':<24} {'PAPERS':>6} {'CHECKED':>8} {'UNRESOLVED':>11} {'RATE':>7}  95% CI"
        f"        {'DEGRADED':>8} {'NO arXiv ID':>11}",
    ]
    for system, s in summary.items():
        rate = "-" if s["rate"] is None else f"{s['rate']:.1%}"
        ci = "-" if s["rate"] is None else f"{s['ci95'][0]:.3f}-{s['ci95'][1]:.3f}"
        lines.append(
            f"  {system:<24} {s['papers']:>6} {s['checked']:>8} {s['unresolved']:>11} "
            f"{rate:>7}  {ci:<13} {s['degraded']:>8} {s['not_checkable']:>11}"
        )
    lines += [
        "",
        "  Only arXiv identifiers are checked. A paper citing none is not",
        "  checkable, and a paper whose lookups all failed is degraded; neither",
        "  counts toward the rate.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None, *, lookup: Lookup | None = None) -> int:
    """``python -m rig.posthoc_audit ROOT``: audit every released paper under ROOT.

    Reaches arXiv through the adapter's cached resolver, about one request every
    3 seconds. ``lookup`` is for tests.
    """
    parser = argparse.ArgumentParser(prog="python -m rig.posthoc_audit", description=main.__doc__)
    parser.add_argument("root", type=Path, help="one folder per system, papers as text")
    parser.add_argument("--cache", default=".cache/arxiv")
    parser.add_argument("--out", type=Path, help="write every paper's result and the summary as JSON")
    args = parser.parse_args(argv)

    if lookup is None:
        from gates.adapters.arxiv import arxiv_lookup

        lookup = arxiv_lookup(cache_dir=args.cache)
    results = audit_tree(args.root, lookup)
    summary = summarise(results)
    print(render(summary))
    if args.out is not None:
        args.out.write_text(
            json.dumps({"summary": summary, "papers": [asdict(r) for r in results]}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
