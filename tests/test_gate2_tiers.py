"""Gate 2's tiers compared, held by test: each one adds what the others cannot.

The per-tier evaluations score each tier on its own corpus. These tests hold the
comparison: tier A alone is blind to every plan defect, tier B adds them without
costing a boundary catch or a false rejection, and tier C turns findings into
fixes rather than declarations. `rig/gate2_tier_comparison.py` prints the table.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rig.gate2_tier_comparison import detection, loop, main  # noqa: E402


@pytest.fixture(scope="module")
def table():
    return detection()


@pytest.fixture(scope="module")
def runs():
    return loop()


def test_tier_a_alone_misses_every_plan_defect(table):
    """Without a declared plan, nothing in tier A can tell 0.01 from 0.001."""
    assert table["plan_divergences"]["A"] == 0
    assert table["unverifiable_fields"]["A"] == 0


def test_tier_b_catches_the_plan_defects_tier_a_cannot(table):
    for name in ("plan_divergences", "unverifiable_fields"):
        assert table[name]["A+B"] == table[name]["n"], name


def test_adding_tier_b_costs_no_boundary_catch(table):
    row = table["boundary_defects"]
    assert row["A"] == row["A+B"] == row["n"]


def test_no_tier_rejects_a_legitimate_run(table):
    """An unverifiable field counts as legitimate here: it must warn, not fail."""
    assert table["false_rejections"]["A"] == 0
    assert table["false_rejections"]["A+B"] == 0


def test_tier_c_turns_findings_into_fixes(runs):
    """One shot, every finding stands declared. The loop fixes what can be fixed
    and declares only what the engineer would not change."""
    assert runs["fixed_one_shot"] == 0
    assert runs["fixed_with_loop"] > runs["fixed_one_shot"]
    assert runs["declared_with_loop"] < runs["declared_one_shot"]
    assert runs["fixed_with_loop"] + runs["declared_with_loop"] == runs["entering"]


def test_tier_c_never_reviews_a_fix_gate_1_rejected(runs):
    assert runs["fixes_gate1_rejected"] >= 1


def test_the_published_comparison(table, runs):
    """The numbers the paper reports. A change that moves them updates
    `docs/research/gate2-tier-comparison.md` in the same commit."""
    assert table == {
        "boundary_defects": {"n": 27, "A": 27, "A+B": 27},
        "plan_divergences": {"n": 12, "A": 0, "A+B": 12},
        "unverifiable_fields": {"n": 6, "A": 0, "A+B": 6},
        "false_rejections": {"n": 35, "A": 0, "A+B": 0},
    }
    assert runs == {
        "runs": 6,
        "entering": 4,
        "fixed_one_shot": 0,
        "fixed_with_loop": 3,
        "declared_one_shot": 4,
        "declared_with_loop": 1,
        "fixes_gate1_rejected": 1,
    }


def test_the_cli_exits_clean(capsys):
    assert main() == 0
    assert "TIER C" in capsys.readouterr().out
