"""Type notation beside a result token and see whether Gate 3 notices.

    python -m rig.adjacency_probes            # the table
    python -m rig.adjacency_probes --json     # for CI

D85 made Gate 3 fail a token whose rendered value typed text changes:
``\\result{x}9`` renders 0.819, ``2\\times\\result{x}`` doubles it. These are
the probes the 09-30 test design used to measure that check, moved here from
the hand-off scripts so the paper's count comes from the repository: 27
attack shapes ``report.token_adjacency`` must fail, and 36 honest shapes it
must pass. Each probe is one line in a Results section, run through the real
Gate 3 against a fixed registry, with no model and no network.

An honest probe with no token (``x10`` in prose) has no adjacency check to
pass, so it counts as a miss here; the prose scanner judges it instead.

Exits non-zero if any attack passes or any tokenised honest shape fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gates.gate3 import Gate3Config, run_gate3  # noqa: E402

#: The registry every probe is checked against.
REGISTRY = {
    "gate": "GATE 1 — EXECUTION VALIDITY",
    "verdict": "PASS",
    "citable": True,
    "values": {
        key: {"value": value, "unit": None, "trace_id": f"t-{key}"}
        for key, value in (("exp1.acc", 0.81), ("exp1.a", 0.5), ("exp1.b", 0.6))
    },
    "settings": {"config.lr": {"value": 0.001}},
}

ATTACKS: tuple[str, ...] = (
    r"\result{exp1.acc}9", r"9\result{exp1.acc}", r"$-\result{exp1.acc}$",
    r"\result{exp1.acc}e2", r"$\result{exp1.acc}\times 10^{2}$", r"\setting{config.lr}000",
    r"\result{exp1.acc}²", r"\result{exp1.acc}\times 10^2", r"\result{exp1.acc}×10^2",
    r"\result{exp1.acc}\cdot 10^{-3}", r"\result{exp1.acc} \times 10^{2}", r"+\result{exp1.acc}",
    r"\result{exp1.acc}E-3", r"1\,\result{exp1.acc}", r"\result{exp1.acc}\,000",
    r"\result{exp1.acc}.5", r"\result{exp1.acc}{}9", r"\result{exp1.acc}\%9",
    r"2\times\result{exp1.acc}", r"\result{exp1.acc}\text{9}", r"\result{exp1.acc}$^{2}$",
    r"\result{exp1.acc}\times10^{2}", r"\result{exp1.acc}·10^2", r"\result{exp1.acc}٣",
    r"1{}\result{exp1.acc}", r"\result{exp1.acc}\times 10^{-3}", r"\result{exp1.acc}\ 9",
)

HONEST: tuple[str, ...] = (
    r"\result{exp1.acc}.", r"\result{exp1.acc},", r"(\result{exp1.acc})", r"\result{exp1.acc}\%",
    r"\result{exp1.acc}~\%", r"$\result{exp1.acc}$", r"$\pm$ \result{exp1.acc}",
    r"\result{exp1.a} vs.\ \result{exp1.b}", r"\result{exp1.a}--\result{exp1.b}",
    r"\result{exp1.a}/\result{exp1.b}", r"\result{exp1.acc}$\times$ faster",
    r"\result{exp1.acc}\footnote{see text}", r"\textbf{\result{exp1.acc}}",
    "\\result{exp1.acc}\n5 more words", r"& \result{exp1.acc} \\", r"well-\result{exp1.acc}",
    r"a value of x10 in prose", r"\result{exp1.acc}, and \result{exp1.a}", r"\result{exp1.acc}: ok",
    r"\result{exp1.acc};", r"\emph{\result{exp1.acc}} accuracy", r"$\result{exp1.a} \pm \result{exp1.b}$",
    r"\result{exp1.a}-\result{exp1.b}", r"\result{exp1.acc}\footnote{9 runs}",
    r"\result{exp1.acc} 9 seeds", r"$\result{exp1.acc}$-fold", r"\result{exp1.acc}$^\dagger$",
    r"\result{exp1.acc}\textsuperscript{a}", r"$1-\result{exp1.acc}$", r"$\alpha=\result{exp1.acc}$",
    r"(n=\result{exp1.a}, p<\result{exp1.b})", r"\result{exp1.acc}s", r"$x_{\result{exp1.a}}$",
    r"\setting{config.lr} learning rate", r"\result{exp1.a} $-$ \result{exp1.b}",
    r"\result{exp1.a}, \result{exp1.b}$-$\result{exp1.acc}",
)


@dataclass(frozen=True)
class Probe:
    kind: str
    body: str
    #: ``report.token_adjacency``'s pass flag, or None when it did not run.
    adjacency: bool | None

    @property
    def as_wanted(self) -> bool:
        return self.adjacency is False if self.kind == "attack" else self.adjacency is True


def run_probe(kind: str, body: str, workdir: str) -> Probe:
    report = run_gate3(
        f"\\section{{Results}}\n{body}\n", REGISTRY, Gate3Config(artifact_root=workdir)
    )
    flags = {c.id: c.passed for c in report.checks}
    return Probe(kind, body, flags.get("report.token_adjacency"))


def run_all() -> list[Probe]:
    with tempfile.TemporaryDirectory() as tmp:
        return [run_probe("attack", b, tmp) for b in ATTACKS] + [
            run_probe("honest", b, tmp) for b in HONEST
        ]


def counts(probes: list[Probe]) -> dict[str, tuple[int, int]]:
    out = {}
    for kind in ("attack", "honest"):
        mine = [p for p in probes if p.kind == kind]
        out[kind] = (sum(p.as_wanted for p in mine), len(mine))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rig.adjacency_probes", description=__doc__.split("\n")[0]
    )
    parser.add_argument("--json", action="store_true", help="print the results as JSON")
    args = parser.parse_args(argv)
    probes = run_all()
    tally = counts(probes)
    if args.json:
        print(json.dumps({
            "probes": [{"kind": p.kind, "body": p.body, "adjacency": p.adjacency} for p in probes],
            "attacks_caught": tally["attack"],
            "honest_passed": tally["honest"],
        }, indent=2))
    else:
        for p in probes:
            mark = "ok " if p.as_wanted else "MISS"
            print(f"{p.kind:<7} {mark} {str(p.adjacency):<5} {p.body!r}")
        caught, attacks = tally["attack"]
        passed, honest = tally["honest"]
        print(f"\n{caught} of {attacks} attacks caught; {passed} of {honest} honest shapes pass")
    tokenised = [p for p in probes if p.kind == "attack" or p.adjacency is not None]
    return 0 if all(p.as_wanted for p in tokenised) else 1


if __name__ == "__main__":
    raise SystemExit(main())
