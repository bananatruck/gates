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
ARCHIVED_RUN = HOST / "results" / "gemini_3_5_flash_run_1"
#: The completed full-workflow ablation: both arms, same task, same models.
ABLATION = HOST / "full_ablation_runs" / "deepseek_common_20260815"
GATED_DIR = ABLATION / "gated" / "research_dir"

#: The like-for-like baseline: the same task and the same models as the gated
#: run, with GATES_GATE1=off. The archived paper is a different topic, so on its
#: own it can only show that the failure happens -- not that Gate 1 is what
#: prevents it. This pair is what makes the comparison an ablation.
#: The first ungated arm died mid-run (a None report reached the ledger); the
#: retry is the one that carried through, so it is what gets audited. Falls back
#: to the original arm if the retry is absent.
_RETRY = HOST / "full_ablation_runs" / (
    "deepseek_common_20260815_ungated_retry") / "ungated" / "research_dir"
UNGATED_DIR = _RETRY if _RETRY.exists() else ABLATION / "ungated" / "research_dir"


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


def audit_archived() -> PaperAudit:
    """The shipped scaffold's own output, on its own topic (SGC/Cora)."""
    paper = ARCHIVED_RUN / "research_dir" / "report.txt"
    code_path = ARCHIVED_RUN / "research_dir" / "src" / "run_experiments.py"
    code = code_path.read_text(errors="replace") if code_path.exists() else ""
    # No registry: the run never recorded anything, which is the finding.
    # No execution capture either -- only a workflow transcript, which
    # `audit` refuses, so nothing here is scored as "printed".
    return audit(paper, code_text=code)


def last_ungated_results(gate_root: Path) -> Path | None:
    """The last `results.json` the gate-off arm's executions left behind."""
    found = [d / "results.json" for d in sorted(gate_root.glob("ungated_*"))
             if (d / "results.json").exists()]
    return found[-1] if found else None


def audit_ungated() -> PaperAudit:
    """Same task and models as the gated run, gate switched off.

    Audited against what its own run recorded on disk -- which is deliberately
    generous, because none of it reached the writing agent. `record_result`
    still works without the gate; what the gate adds is delivery, so the values
    sat in results.json while the writer saw 1,000 characters of training log.
    Crediting the arm for numbers it could not see keeps the comparison
    symmetric and makes the gap that survives the harder claim.
    """
    paper = UNGATED_DIR / "report.txt"
    code_path = UNGATED_DIR / "src" / "run_experiments.py"
    code = code_path.read_text(errors="replace") if code_path.exists() else ""
    registry = last_ungated_results(UNGATED_DIR / "gate_artifacts")
    return audit(paper, registry_path=registry, code_text=code)


def audit_gated() -> PaperAudit:
    paper = GATED_DIR / "report.txt"
    registry = last_passing_registry(GATED_DIR / "gate_artifacts" / "gate1")
    code_path = GATED_DIR / "src" / "run_experiments.py"
    code = code_path.read_text(errors="replace") if code_path.exists() else ""
    return audit(paper, registry_path=registry, code_text=code)


def main() -> None:
    out = {}
    # Order matters for the rendered table: the two arms of the ablation first,
    # since they are the controlled comparison, then the archived run as the
    # independent instance of the same failure on a different topic.
    for name, fn in (("with Gate 1", audit_gated),
                     ("without Gate 1", audit_ungated),
                     ("archived run (different topic, no gate)", audit_archived)):
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
