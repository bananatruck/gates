"""Lay released papers out as run folders so ``rig.judge`` can score them.

    python -m rig.stage_released --papers <release>/papers --tasks <kit>/tasks \\
        --benchmark Hidden-Pitfalls --system "Released system" --model <model> \\
        --out <runs-rescore>

Run ``python -m rig.judge --runs <runs-rescore> ...`` to score the staged papers.

Each task folder holds either ``<task>.pdf`` or ``paper.md``.
It can also hold ``experiments/*.py``.
The stager writes ``<out>/<benchmark>/<task>/<system-slug>/released/seed0/``.

``manifest.json``
    The task, model, source format, source hash, task hash, text extractor, and
    ``code_files`` relative to ``experiments/``.
``paper/report.txt``
    The text sent to the judge.
    ``paper/source.pdf`` or ``paper/source.md`` preserves the released source.
``code/``
    The released experiment scripts, only when ``code_files`` is non-empty.

A task with no task text or paper is named and skipped.
Whitespace-only text is treated as empty and skipped with nothing written.
Failed extraction or reading is skipped the same way.
A system name that slugs to an empty path is refused.
Two different systems that share a slug refuse the second with both names in the note.
A folder already staged is never overwritten.

PDF text comes from poppler's ``pdftotext -layout``.
Markdown text is read directly and records ``source markdown`` as its extractor.
``paper/collect.py`` skips ``phase: rescore``, so a released paper is never one of our levels.

Stdlib only, no model.
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


def _experiment_scripts(source: Path) -> list[Path]:
    root = source / "experiments"
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def _run_folder(out: Path, benchmark: str, task: str, system: str) -> Path:
    return out / benchmark / task / _slug(system) / "released" / "seed0"


def _folder_skip_note(folder: Path, task: str, system: str) -> str:
    manifest_path = folder / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("system") == system:
            return f"{task}: already staged at {folder}"
        other = existing.get("system", "?")
        slug = _slug(system)
        return (f"{task}: slug clash between {other!r} and {system!r} "
                f"(both map to {slug}) at {folder}")
    return f"{task}: already staged at {folder}"


def stage_all(
    papers: Path,
    *,
    tasks: Path,
    out: Path,
    benchmark: str = "MLR-Bench",
    system: str,
    model: str,
    extract: Callable[[Path], str] = pdftotext,
    extractor: str | None = None,
) -> tuple[list[Path], list[str]]:
    """Stage every task folder under ``papers``. Returns the folders and a note per skip."""
    slug = _slug(system)
    staged: list[Path] = []
    skipped: list[str] = []
    for source in sorted(p for p in papers.iterdir() if p.is_dir()):
        task = source.name
        task_file = tasks / f"{task}.md"
        pdf = source / f"{task}.pdf"
        markdown = source / "paper.md"
        if not task_file.is_file():
            skipped.append(f"{task}: no task text at {task_file}")
            continue
        if pdf.is_file():
            source_paper = pdf
            source_format = "pdf"
        elif markdown.is_file():
            source_paper = markdown
            source_format = "markdown"
        else:
            skipped.append(f"{task}: no pdf at {pdf} or Markdown paper at {markdown}")
            continue
        if not slug:
            skipped.append(f"{task}: system {system!r} slugs to an empty path")
            continue
        folder = _run_folder(out, benchmark, task, system)
        if folder.exists():
            skipped.append(_folder_skip_note(folder, task, system))
            continue
        if source_format == "pdf":
            used_extractor = extractor or default_extractor()
            try:
                report = extract(source_paper)
            except Exception as exc:
                skipped.append(f"{task}: extraction failed ({exc})")
                continue
        else:
            used_extractor = "source markdown"
            try:
                report = source_paper.read_text(encoding="utf-8")
            except Exception as exc:
                skipped.append(f"{task}: reading Markdown failed ({exc})")
                continue
        if not report.strip():
            skipped.append(f"{task}: empty paper text from {source_paper}")
            continue
        scripts = _experiment_scripts(source)
        code_files = [str(p.relative_to(source / "experiments")) for p in scripts]
        (folder / "paper").mkdir(parents=True)
        source_name = "source.pdf" if source_format == "pdf" else "source.md"
        shutil.copyfile(source_paper, folder / "paper" / source_name)
        (folder / "paper" / "report.txt").write_text(report, encoding="utf-8")
        if code_files:
            for script in scripts:
                rel = script.relative_to(source / "experiments")
                dest = folder / "code" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(script, dest)
        source_hash = hashlib.sha256(source_paper.read_bytes()).hexdigest()
        manifest = {
            "benchmark": benchmark,
            "task": task,
            "system": system,
            "model": model,
            "phase": "rescore",
            "status": "released",
            "paper_present": True,
            "task_file_sha256": hashlib.sha256(task_file.read_bytes()).hexdigest(),
            "source_paper": source_paper.name,
            "source_paper_sha256": source_hash,
            "source_format": source_format,
            "text_extractor": used_extractor,
            "source": str(source),
            "code_files": code_files,
        }
        if source_format == "pdf":
            manifest["source_pdf_sha256"] = source_hash
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staged.append(folder)
    return staged, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rig.stage_released",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--papers", type=Path, required=True, help="one folder per task")
    parser.add_argument("--tasks", type=Path, required=True, help="one <task>.md file per paper")
    parser.add_argument("--benchmark", default="MLR-Bench")
    parser.add_argument("--system", required=True)
    parser.add_argument("--model", required=True, help="the model the released system used")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    has_pdf = any(
        (source / f"{source.name}.pdf").is_file()
        for source in args.papers.iterdir()
        if source.is_dir()
    )
    if has_pdf and not pdftotext_available():
        print("pdftotext (poppler) is not installed; nothing staged", file=sys.stderr)
        return 2
    staged, skipped = stage_all(args.papers, tasks=args.tasks, out=args.out,
                                benchmark=args.benchmark,
                                system=args.system, model=args.model)
    print(f"{len(staged)} papers staged under {args.out}")
    for note in skipped:
        print(f"  skipped {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
