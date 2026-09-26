"""paper/mechanism.csv against the rigs that produced it.

Figure 8 and the README draw these rows as measured. A row a rig no longer
reproduces is a published number that has quietly stopped being true, so each
rig row is recomputed here. The Gate 1 rows come from the signed 08-15 campaign,
which nothing re-runs, and are checked only for their source.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from rig import gate2_tier_a_eval as tier_a
from rig import gate2_tier_b_eval as tier_b
from rig.gate3_scanner_miss import measure

REPO = Path(__file__).resolve().parents[1]


def published() -> dict[str, tuple[int, int, str]]:
    with open(REPO / "paper" / "mechanism.csv", newline="") as f:
        return {r["id"]: (int(r["k"]), int(r["n"]), r["source"]) for r in csv.DictReader(f)}


def test_tier_a_rows_are_what_the_rig_measures():
    counts = Counter(tier_a.run(c)["outcome"] for c in tier_a.CASES)
    rows = published()
    assert rows["g2a_detect"][:2] == (counts["TP"], counts["TP"] + counts["FN"])
    assert rows["g2a_fpr"][:2] == (counts["FP"], counts["FP"] + counts["TN"])


def test_tier_b_rows_are_what_the_rig_measures():
    results = [tier_b.run(c) for c in tier_b.CASES]
    d = Counter(r["divergence"] for r in results)
    t = Counter(r["traceability"] for r in results)
    rows = published()
    assert rows["g2b_detect"][:2] == (d["TP"], d["TP"] + d["FN"])
    assert rows["g2b_fpr"][:2] == (d["FP"], d["FP"] + d["TN"])
    assert rows["g2b_trace"][:2] == (t["TP"], t["TP"] + t["FN"])
    assert rows["g2b_trace_fpr"][:2] == (t["FP"], t["FP"] + t["TN"])


def test_the_gate_3_row_is_what_the_scanner_rig_measures():
    arms = [measure("gated"), measure("ungated")]
    detected = sum(a.detected for a in arms)
    claims = sum(a.claims for a in arms)
    assert published()["g3_detect"][:2] == (detected, claims)


def test_the_gate_1_rows_name_the_signed_campaign():
    rows = published()
    for key in ("g1_delivered_on", "g1_delivered_off", "g1_traced_on", "g1_traced_off"):
        assert rows[key][2].startswith("reports/finalized-report-and-results")
    assert (REPO / "reports" / "finalized-report-and-results").is_dir()
