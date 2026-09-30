"""Lay released papers out as run folders, so rig.judge re-scores them (D70 item 3).

    python -m rig.stage_released --papers <mlrbench>/ai_scientist_v2_papers/o4-mini \\
        --tasks <kit>/mlrbench/tasks --system "AI Scientist v2" --model o4-mini \\
        --out ~/gates-runs/runs-rescore

Then, on machine A, ``python -m rig.judge --runs ~/gates-runs/runs-rescore ...``
exactly as for our own runs, so the released papers meet D63's judges and
MLR-Bench's prompts with nothing else changed.

Each task folder holds ``<task>.pdf`` and ``experiments/*.py``, as MLR-Bench
released AI Scientist v2's papers at ``f728d57``. The stager writes, per task,
``<out>/MLR-Bench/<task>/<system>/released/seed0/`` with:

``manifest.json``
    task, model, ``phase: rescore``, the task text's hash (which ``rig.judge``
    checks), the PDF's hash, and the text extractor and its version;
``paper/report.txt``
    the PDF's text; ``paper/source.pdf`` beside it;
``code/``
    the released experiment scripts.

One difference from MLR-Bench, stated rather than hidden: its judge reads a
PDF through ``pymupdf4llm``'s Markdown, which the host's environment does not
carry and ``rig.judge`` stubs out. The text here comes from poppler's
``pdftotext -layout``, recorded in each manifest, so a score can differ from
one on MLR-Bench's extraction. ``paper/collect.py`` skips ``phase: rescore``:
a released paper is never a level of ours.

A task with no task text or no PDF is named and skipped; a folder already
staged is never overwritten. Stdlib only, no model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path


def pdftotext_available() -> bool:
    return shutil.which("pdftotext") is not None


def default_extractor() -> str:
    """``pdftotext -layout`` and the version poppler reports, for the manifest."""
    done = subprocess.run(["pdftotext", "-v"], capture_output=True, text=True, check=False)
    version = (done.stderr or done.stdout).splitlines()[0] if (done.stderr or done.stdout) else ""
    return f"pdftotext -layout ({version.strip()})"


def pdftotext(pdf: Path) -> str:
    done = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                          capture_output=True, text=True, check=True)
    return done.stdout


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")


def stage_all(
    papers: Path,
    *,
    tasks: Path,
    out: Path,
    system: str,
    model: str,
    extract: Callable[[Path], str] = pdftotext,
    extractor: str | None = None,
) -> tuple[list[Path], list[str]]:
    """Stage every task folder under ``papers``. Returns the folders and a note per skip."""
    extractor = extractor or default_extractor()
    staged: list[Path] = []
    skipped: list[str] = []
    for source in sorted(p for p in papers.iterdir() if p.is_dir()):
        task = source.name
        task_file = tasks / f"{task}.md"
        pdf = source / f"{task}.pdf"
        if not task_file.is_file():
            skipped.append(f"{task}: no task text at {task_file}")
            continue
        if not pdf.is_file():
            skipped.append(f"{task}: no pdf at {pdf}")
            continue
        folder = out / "MLR-Bench" / task / _slug(system) / "released" / "seed0"
        if folder.exists():
            skipped.append(f"{task}: already staged at {folder}")
            continue
        (folder / "paper").mkdir(parents=True)
        (folder / "code").mkdir()
        shutil.copyfile(pdf, folder / "paper" / "source.pdf")
        (folder / "paper" / "report.txt").write_text(extract(pdf), encoding="utf-8")
        for script in sorted((source / "experiments").glob("*.py")):
            shutil.copyfile(script, folder / "code" / script.name)
        manifest = {
            "benchmark": "MLR-Bench",
            "task": task,
            "system": system,
            "model": model,
            "phase": "rescore",
            "status": "released",
            "paper_present": True,
            "task_file_sha256": hashlib.sha256(task_file.read_bytes()).hexdigest(),
            "source_pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
            "text_extractor": extractor,
            "source": str(source),
        }
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staged.append(folder)
    return staged, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rig.stage_released",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--papers", type=Path, required=True, help="one folder per task")
    parser.add_argument("--tasks", type=Path, required=True, help="the pinned MLR-Bench tasks/")
    parser.add_argument("--system", required=True)
    parser.add_argument("--model", required=True, help="the model the released system used")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not pdftotext_available():
        print("pdftotext (poppler) is not installed; nothing staged", file=sys.stderr)
        return 2
    staged, skipped = stage_all(args.papers, tasks=args.tasks, out=args.out,
                                system=args.system, model=args.model)
    print(f"{len(staged)} papers staged under {args.out}")
    for note in skipped:
        print(f"  skipped {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
