"""Put both generated papers through the same audit and record the result.

One paper was written by Agent Laboratory as shipped; the other by the same
scaffold with Gate 1 in the loop. The comparison is only worth anything if
neither gets a procedure the other did not, so both go through
`rig.paper_audit.audit` with the same claim filter and the same tolerance. The
only asymmetry is the one being measured: the gated run has a registry to check
against, and the ungated run has nothing, because it never recorded anything.

Which registry the gated paper is checked against matters. The paper is written
from the code the solver finally accepted, so the registry that backs it is the
last *passing* attempt's -- not the last attempt's, which may be a rejected one,
and not a union across attempts, which would let a number sourced in an
abandoned attempt vouch for a claim the accepted code never produced.
"""

from __future__ import annotations

import json
from pathlib import Path

from rig.paper_audit import PaperAudit, audit, render

HOST = Path("/home/kesh/AgentLaboratory-Gemini")
UNGATED_RUN = HOST / "results" / "gemini_3_5_flash_run_1"
GATED_DIR = HOST / "research_dir"


def last_passing_registry(gate_root: Path) -> Path | None:
    """The registry behind the code the solver actually accepted."""
    best = None
    for attempt in sorted(gate_root.glob("attempt_*")):
        report = attempt / "gate1_report.json"
        registry = attempt / "registry.json"
        if not report.exists() or not registry.exists():
            continue
        try:
            verdict = json.loads(report.read_text()).get("verdict")
        except json.JSONDecodeError:
            continue
        if verdict == "PASS":
            best = registry
    return best


def audit_ungated() -> PaperAudit:
    paper = UNGATED_RUN / "research_dir" / "report.txt"
    code_path = UNGATED_RUN / "research_dir" / "src" / "run_experiments.py"
    code = code_path.read_text(errors="replace") if code_path.exists() else ""
    # No registry: the run never recorded anything, which is the finding.
    # No execution capture either -- only a workflow transcript, which
    # `audit` refuses, so nothing here is scored as "printed".
    return audit(paper, code_text=code)


def audit_gated() -> PaperAudit:
    paper = GATED_DIR / "report.txt"
    registry = last_passing_registry(GATED_DIR / "gate_artifacts" / "gate1")
    code_path = GATED_DIR / "src" / "run_experiments.py"
    code = code_path.read_text(errors="replace") if code_path.exists() else ""
    return audit(paper, registry_path=registry, code_text=code)


def main() -> None:
    out = {}
    for name, fn in (("without Gate 1", audit_ungated), ("with Gate 1", audit_gated)):
        result = fn()
        out[name] = result.to_dict()
        print(f"--- {name} ---")
        print(render(result))
        print()

    dest = HOST / "ablation_runs" / "paper_audit.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
