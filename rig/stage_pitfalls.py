"""Lay Hidden Pitfalls' released Agent Laboratory outputs out as run folders.

    python -m rig.stage_pitfalls --release <AIScientistPitfalls> --model o3-mini \\
        --out ~/gates-runs/pitfalls-B

Then audit them with ``python -m rig.pitfalls_audit --runs <out> --mlrbench <out> ...``:
the stager writes each task text to ``<out>/tasks/``, which is the folder the
auditor reads task texts from.

The release is https://github.com/niharshah/AIScientistPitfalls at ``2725bda``
(Luo, Kasirzadeh and Shah, arXiv:2509.08713). Its Agent Laboratory outputs sit
under ``AgentLaboratory/generated_research/<pitfall>/.../`` with ``report.txt``
(the paper as text), ``src/*.py`` and ``src/experiment_output.log``. The model is
``o3-mini``, from ``AgentLaboratory/experiment_configs/SPR_agentlab.yaml``.

Task text per pitfall, from ``SPR Task/``:

* ``BenchmarkIssue`` and ``DataLeakage``: that pitfall's ``SPR.md``;
* ``MetricMisuse``: ``color-first/SPR.md`` or ``shape-first/SPR.md``, as the
  output's ``color first`` or ``shape first`` folder names. The ``*-flip``
  folders above them are dataset variants and share the task text.

``PostHocSelection`` released candidate ``project*.txt`` files, not papers, and
has no ``SPR.md``; it is not staged. An output with no ``report.txt`` (a run
that crashed before writing) is named and skipped.

Each output becomes ``<out>/Hidden-Pitfalls/<task>/<system-slug>/released/<path-slug>/``
with ``manifest.json``, ``paper/report.txt``, ``code/`` and ``agent_log.txt``.
A folder already staged is never overwritten; a task text that differs from the
one already in ``<out>/tasks/`` is refused, so one task name never covers two
texts. ``paper/collect.py`` skips ``phase: rescore``.

Stdlib only, no model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from rig.stage_released import _slug

BENCHMARK = "Hidden-Pitfalls"
SYSTEM = "Agent Laboratory"
LOG = Path("src") / "experiment_output.log"

TaskFor = Callable[[Path], "tuple[str, Path] | None"]


def agentlab_task_for(spr_tasks: Path) -> TaskFor:
    """The task name and task text for an output, from its path under ``generated_research``."""

    def task_for(rel: Path) -> tuple[str, Path] | None:
        pitfall = rel.parts[0]
        if pitfall in ("BenchmarkIssue", "DataLeakage"):
            return pitfall, spr_tasks / pitfall / "SPR.md"
        if pitfall == "MetricMisuse":
            for order in ("color first", "shape first"):
                if order in rel.parts:
                    variant = order.replace(" ", "-")
                    return f"MetricMisuse-{variant}", spr_tasks / "MetricMisuse" / variant / "SPR.md"
        return None

    return task_for


def _outputs(research: Path) -> list[Path]:
    """Every folder that holds a paper or the source of a run."""
    found = {p.parent for p in research.rglob("report.txt")}
    found |= {p.parent.parent for p in research.rglob(str(LOG))}
    return sorted(found)


def _copy_task(task_file: Path, dest: Path) -> bool:
    """Put the task text at ``dest``. False when ``dest`` holds a different text."""
    text = task_file.read_bytes()
    if dest.is_file():
        return dest.read_bytes() == text
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(text)
    return True


def stage_all(
    research: Path,
    *,
    task_for: TaskFor,
    out: Path,
    system: str,
    model: str,
) -> tuple[list[Path], list[str]]:
    """Stage every output under ``research``. Returns the folders and a note per skip."""
    staged: list[Path] = []
    skipped: list[str] = []
    for source in _outputs(research):
        rel = source.relative_to(research)
        report = source / "report.txt"
        if not report.is_file():
            skipped.append(f"{rel}: no report.txt, so no paper to audit")
            continue
        task = task_for(rel)
        if task is None:
            skipped.append(f"{rel}: no task text for this pitfall")
            continue
        name, task_file = task
        if not task_file.is_file():
            skipped.append(f"{rel}: no task text at {task_file}")
            continue
        paper = report.read_bytes()
        if not paper.strip():
            skipped.append(f"{rel}: empty report.txt")
            continue
        folder = out / BENCHMARK / name / _slug(system) / "released" / _slug("/".join(rel.parts[1:]))
        if folder.exists():
            skipped.append(f"{rel}: already staged at {folder}")
            continue
        tasks_copy = out / "tasks" / f"{name}.md"
        if not _copy_task(task_file, tasks_copy):
            skipped.append(f"{rel}: {tasks_copy} already holds a different task text")
            continue

        src = source / "src"
        scripts = sorted(p for p in src.rglob("*.py") if p.is_file()) if src.is_dir() else []
        log = source / LOG
        manifest = {
            "benchmark": BENCHMARK,
            "task": name,
            "system": system,
            "model": model,
            "phase": "rescore",
            "status": "released",
            "paper_present": True,
            "task_file_sha256": hashlib.sha256(task_file.read_bytes()).hexdigest(),
            "source_paper": "report.txt",
            "source_paper_sha256": hashlib.sha256(paper).hexdigest(),
            "source_format": "text",
            "text_extractor": "source text",
            "source": str(source),
            "code_files": [str(p.relative_to(src)) for p in scripts],
            "agent_log": str(LOG) if log.is_file() else None,
        }
        # Built beside its final place and renamed, so a failure leaves no half-staged folder.
        folder.parent.mkdir(parents=True, exist_ok=True)
        building = Path(tempfile.mkdtemp(prefix=f".{folder.name}.", dir=folder.parent))
        try:
            (building / "paper").mkdir()
            (building / "paper" / "report.txt").write_bytes(paper)
            for script in scripts:
                dest = building / "code" / script.relative_to(src)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(script, dest)
            if log.is_file():
                shutil.copyfile(log, building / "agent_log.txt")
            (building / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            building.rename(folder)
        except BaseException:
            shutil.rmtree(building, ignore_errors=True)
            raise
        staged.append(folder)
    return staged, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rig.stage_pitfalls",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--release", type=Path, required=True,
                        help="a checkout of niharshah/AIScientistPitfalls")
    parser.add_argument("--model", required=True, help="the model the released system used")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    staged, skipped = stage_all(
        args.release / "AgentLaboratory" / "generated_research",
        task_for=agentlab_task_for(args.release / "SPR Task"),
        out=args.out, system=SYSTEM, model=args.model,
    )
    print(f"{len(staged)} papers staged under {args.out}")
    for note in skipped:
        print(f"  skipped {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
