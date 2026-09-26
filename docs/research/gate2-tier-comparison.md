# Gate 2 tiers compared: what each one adds

Branch `feature/gate2-feedback-loop`.

Reproduce with `python -m rig.gate2_tier_comparison`.
Held by `tests/test_gate2_tiers.py`.

## A against A+B

Every case in the tier A corpus (45) and the tier B corpus (29) runs twice: with the plan withheld (tier A alone) and with it (A+B).
There is no B-alone column.
Ranges run on every registry, so no host can switch tier A off.

| Class | n | A | A+B |
|---|---|---|---|
| Boundary defects caught | 27 | 27 | 27 |
| Plan divergences caught | 12 | 0 | 12 |
| Unverifiable plan fields flagged | 6 | 0 | 6 |
| Legitimate runs rejected | 35 | 0 | 0 |

Tier A alone is blind to every plan defect.
A learning rate of 0.01 against a declared 0.001 is a legal number, so no range or relation can see it.
Tier B adds all 18 without losing a boundary catch or rejecting a legitimate run.
The 35 legitimate runs include the 6 unverifiable fields, which must warn and never fail.

These are coverage figures over cases we wrote, not skill figures.
The Wilson intervals in the two per-tier documents apply here unchanged.

## A+B against A+B+C

Tier C adds no check.
It is compared on what happens after detection, over the six loop scenarios in `rig/gate2_scenarios.py`, four of which have a finding on first review.
One shot means the first review is final.

| Outcome | A+B | A+B+C |
|---|---|---|
| Findings fixed before writing | 0 | 3 |
| Findings declared to the writer | 4 | 1 |

The one declared finding is the scenario where the engineer never changes the learning rate: Gate 2 proceeds with it declared, as designed.

Every fix runs under Gate 1 before Gate 2 reviews it (F12).
In the `hand-typed-fix` scenario the engineer types the corrected speedup into the `record_result` call.
Gate 1 rejects it, Gate 2 never sees it, and the Gate 2 budget still has the turn the recomputed run passes on.

## Limits

Six scripted scenarios show the loop closes.
They are not a resolution rate for a real engineer.
That rate is spec M5 over MLR-Bench runs (F9), and `Ledger.loop_summary()` computes it from the same ledger rows these scenarios write.
