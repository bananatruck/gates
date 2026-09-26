"""The post-hoc audit of released papers, run with a dict-backed resolver."""

from __future__ import annotations

import json

import pytest

from gates.schema import PaperRecord
from rig.posthoc_audit import audit_tree, main, summarise

KNOWN = {
    "1902.07153": PaperRecord(identifier="1902.07153", title="Simplifying GCNs", year=2019),
    "1706.03762": PaperRecord(identifier="1706.03762", title="Attention", year=2017),
}


def lookup(identifier):
    if identifier == "2401.00001":
        raise OSError("arXiv unreachable")
    return KNOWN.get(identifier)


@pytest.fixture
def papers(tmp_path):
    root = tmp_path / "papers"
    a, b = root / "system-a", root / "system-b"
    a.mkdir(parents=True)
    b.mkdir()
    (a / "clean.tex").write_text("As in arXiv:1902.07153v2 and arXiv:1706.03762.\n")
    (a / "invented.tex").write_text("Following arXiv:2599.99999 and arXiv:1902.07153.\n")
    (a / "no_ids.md").write_text("As shown by Smith et al. (2020).\n")
    (a / "figure.png").write_bytes(b"\x89PNG")
    (b / "outage.txt").write_text("See arXiv:2401.00001.\n")
    (b / "clean.txt").write_text("See arXiv:1706.03762.\n")
    return root


def test_each_paper_gets_exactly_one_outcome(papers):
    results = {(r.system, r.paper): r for r in audit_tree(papers, lookup)}
    assert set(results) == {
        ("system-a", "clean.tex"), ("system-a", "invented.tex"),
        ("system-a", "no_ids.md"), ("system-b", "outage.txt"), ("system-b", "clean.txt"),
    }
    assert results["system-a", "invented.tex"].outcome == "unresolved"
    assert results["system-a", "invented.tex"].unresolved == ["2599.99999"]
    assert results["system-a", "clean.tex"].outcome == "resolved"
    assert results["system-a", "no_ids.md"].outcome == "not_checkable"
    assert results["system-b", "outage.txt"].outcome == "degraded"


def test_the_rate_counts_only_papers_that_were_actually_checked(papers):
    summary = summarise(audit_tree(papers, lookup))
    a, b = summary["system-a"], summary["system-b"]
    assert (a["papers"], a["checked"], a["unresolved"], a["not_checkable"]) == (3, 2, 1, 1)
    assert a["rate"] == 0.5
    assert (b["checked"], b["degraded"], b["rate"]) == (1, 1, 0.0)


def test_the_cli_writes_every_paper_and_the_summary(papers, tmp_path, capsys):
    out = tmp_path / "audit.json"
    assert main([str(papers), "--out", str(out)], lookup=lookup) == 0
    assert "system-a" in capsys.readouterr().out
    written = json.loads(out.read_text())
    assert written["summary"]["system-a"]["unresolved"] == 1
    assert len(written["papers"]) == 5
