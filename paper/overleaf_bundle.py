"""Pack the draft into one zip that Overleaf can compile as uploaded.

    python3 paper/overleaf_bundle.py      # writes .cache/overleaf/gates_aamas2027.zip

The zip holds main.tex, refs.bib, the AAMAS class and bibliography style, the
licence badge, and every figure main.tex includes, under figures/. main.tex
sets \\graphicspath{{figures/}{../figures/}}, so the same source compiles in the
repository and in Overleaf without edits. Stdlib only.
"""

import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRAFT = HERE / "draft"
FIGURES = HERE / "figures"
ROOT_FILES = ("main.tex", "refs.bib", "aamas.cls", "ACM-Reference-Format.bst", "by.pdf")
INCLUDE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def included_figures(tex: str) -> list[str]:
    """Names of the figures main.tex includes from figures/, in order, once each."""
    names: list[str] = []
    for name in INCLUDE.findall(tex):
        if name == "by" or name in names:
            continue
        names.append(name)
    return names


def build(out: Path) -> Path:
    tex = (DRAFT / "main.tex").read_text(encoding="utf-8")
    missing = [f for f in ROOT_FILES if not (DRAFT / f).is_file()]
    figures = included_figures(tex)
    missing += [f"figures/{n}.pdf" for n in figures if not (FIGURES / f"{n}.pdf").is_file()]
    if missing:
        raise SystemExit("missing: " + ", ".join(missing))
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in ROOT_FILES:
            bundle.write(DRAFT / name, name)
        for name in figures:
            bundle.write(FIGURES / f"{name}.pdf", f"figures/{name}.pdf")
    return out


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / ".cache/overleaf/gates_aamas2027.zip"
    path = build(target)
    with zipfile.ZipFile(path) as bundle:
        print(f"wrote {path}:")
        for info in bundle.infolist():
            print(f"  {info.filename}  {info.file_size} bytes")
