"""rig/stats.py against values worked out by hand.

These functions produce the paper's significance claims, so each is checked on
a case small enough to verify on paper.
"""

from __future__ import annotations

import math

import pytest

from rig import corpus, gate2_tier_a_eval, gate2_tier_b_eval
from rig.stats import (
    holm,
    mcnemar_exact,
    mcnemar_pairs_needed,
    non_inferior,
    paired_bootstrap_ci,
    wilson,
)


def test_every_rig_uses_the_one_wilson():
    assert corpus.wilson is wilson
    assert gate2_tier_a_eval.wilson is wilson
    assert gate2_tier_b_eval.wilson is wilson


def test_wilson_on_a_perfect_score_still_leaves_room_below():
    """27/27, tier A's recall: worked by hand, the lower bound is 0.8754."""
    lo, hi = wilson(27, 27)
    assert round(lo, 4) == 0.8754 and hi == 1.0


def test_mcnemar_is_a_fair_coin_over_the_discordant_pairs():
    # 0 of 10 discordant pairs going the other way: 2 * 0.5**10.
    assert math.isclose(mcnemar_exact(10, 0), 2 / 1024)
    # 1 of 10: 2 * (1 + 10) / 1024.
    assert math.isclose(mcnemar_exact(9, 1), 22 / 1024)
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 0) == 1.0
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


def test_holm_steps_down_and_stays_monotone():
    out = holm({"core": 0.01, "mlr": 0.04, "bad": 0.03})
    assert math.isclose(out["core"]["p_adjusted"], 0.03)
    assert math.isclose(out["bad"]["p_adjusted"], 0.06)
    # The largest p is multiplied by 1, giving 0.04, but monotone means it can
    # never fall below the 0.06 before it.
    assert math.isclose(out["mlr"]["p_adjusted"], 0.06)
    assert [out[k]["reject"] for k in ("core", "bad", "mlr")] == [True, False, False]


def test_bootstrap_is_exact_on_constant_differences_and_seeded():
    assert paired_bootstrap_ci([0.25] * 12) == (0.25, 0.25, 0.25)
    data = [-1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 0.0, 1.5]
    assert paired_bootstrap_ci(data, seed=7) == paired_bootstrap_ci(data, seed=7)
    mean, lo, hi = paired_bootstrap_ci(data, reps=2000)
    assert lo < mean < hi
    with pytest.raises(ValueError):
        paired_bootstrap_ci([])


def test_non_inferiority_is_a_bound_on_the_lower_limit():
    assert non_inferior(-0.4, 0.5)
    assert not non_inferior(-0.5, 0.5)
    with pytest.raises(ValueError):
        non_inferior(0.0, 0.0)


def test_pairs_needed_matches_connors_formula():
    # 30% of pairs discordant one way, 5% the other: worked by hand from
    # n = (1.96 * sqrt(0.35) + 0.8416 * sqrt(0.35 - 0.0625))**2 / 0.0625.
    assert mcnemar_pairs_needed(0.30, 0.05) == 42
    with pytest.raises(ValueError):
        mcnemar_pairs_needed(0.1, 0.1)
