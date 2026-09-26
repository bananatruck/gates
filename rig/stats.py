"""The statistics the rigs and the evaluation report, defined once.

Stdlib only, like ``gates/``, so a result can be recomputed on a bare
interpreter. Every function here is the plain textbook form: the numbers go into
a paper, and a reviewer should be able to check them by hand.

The significance design these serve is in ``paper/PLAN.md`` §9, fixed before
any benchmark run: an exact McNemar test per benchmark on the paired integrity
outcome, Holm across benchmarks, and a paired bootstrap interval for the task
score's non-inferiority.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from statistics import NormalDist


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion.

    Used instead of the normal approximation, which is unusable at the sample
    sizes the rigs measure. No trials means no information: the whole interval.
    """
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts.

    ``b`` counts pairs where only the first arm had the outcome, ``c`` pairs
    where only the second did. Concordant pairs carry no information about the
    difference, so they do not appear. Under the null each discordant pair is a
    fair coin, so this is a binomial test of ``min(b, c)`` in ``b + c`` at 0.5.
    """
    if b < 0 or c < 0:
        raise ValueError("discordant counts cannot be negative")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def holm(pvalues: Mapping[str, float], alpha: float = 0.05) -> dict[str, dict[str, float | bool]]:
    """Holm's step-down correction: each name's adjusted p and whether it rejects.

    Adjusted values are made monotone, so a hypothesis never rejects while one
    with a smaller raw p does not.
    """
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    m = len(ordered)
    out: dict[str, dict[str, float | bool]] = {}
    running = 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        out[name] = {"p": p, "p_adjusted": running, "reject": running <= alpha}
    return out


def paired_bootstrap_ci(
    differences: Sequence[float],
    *,
    reps: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean paired difference and its percentile bootstrap interval.

    ``differences`` holds one treated-minus-control value per task and seed.
    Resampling the pairs, not the arms, keeps the pairing that makes the
    comparison fair. Seeded, so a published interval can be recomputed exactly.
    """
    n = len(differences)
    if n == 0:
        raise ValueError("no paired differences to bootstrap")
    rng = random.Random(seed)
    means = sorted(
        sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps)
    )
    tail = (1 - level) / 2
    lo = means[max(0, math.floor(tail * reps))]
    hi = means[min(reps - 1, math.ceil((1 - tail) * reps) - 1)]
    return (sum(differences) / n, lo, hi)


def non_inferior(ci_low: float, margin: float) -> bool:
    """Treated is no worse than control by more than ``margin``.

    ``margin`` is positive and fixed before the run; the lower bound of the
    treated-minus-control interval must stay above its negative.
    """
    if margin <= 0:
        raise ValueError("a non-inferiority margin is a positive distance")
    return ci_low > -margin


def mcnemar_pairs_needed(
    p_first_only: float,
    p_second_only: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Pairs a McNemar test needs to detect this discordance (Connor, 1987).

    The inputs are the expected shares of pairs where only one arm has the
    outcome, taken from a pilot. The normal approximation is the standard one
    for planning; the test itself is run exact.
    """
    discordant = p_first_only + p_second_only
    delta = p_first_only - p_second_only
    if delta == 0:
        raise ValueError("equal discordance has no effect to detect")
    if discordant <= 0 or discordant > 1:
        raise ValueError("discordant share must be in (0, 1]")
    z = NormalDist()
    z_alpha = z.inv_cdf(1 - alpha / 2)
    z_beta = z.inv_cdf(power)
    need = (z_alpha * math.sqrt(discordant) + z_beta * math.sqrt(discordant - delta**2)) ** 2
    return math.ceil(need / delta**2)
