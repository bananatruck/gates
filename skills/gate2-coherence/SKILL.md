---
name: gate2-coherence
description: Wire Gate 2 (source-result coherence) after Gate 1, so a verified run is also checked against the ranges, relations and methodology its plan declared. Use when installing Gate 2, or when a scaffold's results can execute cleanly yet contradict the plan.
---

# Gate 2: source-result coherence

Gate 2 reviews the registry of a run Gate 1 passed, against what the plan declared.
It opens at `GATES_LEVEL=2`.
Install `gate1-execution` first: every revision runs under Gate 1 before Gate 2 sees it, so a fix cannot reach Gate 2 without running.

**Needs:** the `GatedExecution` of a run Gate 1 passed, and Gate 1's context.

**Hands on:** a `ReviewOutcome`. `registry` is what the writer cites, `declared` is the limitations the manuscript must state, and `run` is the code and evidence the writer describes.

## Declare

| Declaration | Feeds | Why the host must say |
|---|---|---|
| `ranges`, `relations` | tier A | a metric's name never implies its range |
| `plan_fields` | tier B | never parsed from plan prose; `gates/` never reads a plan |
| `sources` with `lit_review` | reference bands | a band may only come from a paper the host fetched |

A tier with no declaration does not run, and the report carries no row for it.

## Wire it

1. **Context.** Build it with your adapter's `make_review_context(...)`, carrying the declarations.
2. **Call site.** After the final Gate 1 run, when it passed and `gate_level() >= 2`: `outcome = review_loop(review_ctx, revise, gate1=gate1_ctx, first=final)`.
3. **`revise(feedback)`** returns the engineer's next version of the code, or `None` to stop. If the host's solver ranks entries by a reward score:
   - void the reward score of the entry it last returned, so a fix the reward model likes less still replaces it;
   - append the feedback to the solver's notes as text, never as a list;
   - return `None` when the solver produced nothing new, since resubmitting spends a turn to hear the same rejection.
4. **Hand on.** Give the writer `outcome.run.code`, `outcome.run.evidence_bundle` and `outcome.declared`, not the code the phase started with: once a revision is reviewed, the registry the writer cites came from that run.
5. **Spent budget.** The loop **proceeds** with the discrepancies declared. A genuine novel result must not be blocked forever.

Done when, at `GATES_LEVEL=2`, a registry that breaks a declared range goes back to the engineer, and the writer receives the reviewed run with its limitations appended.

## Prove it

- `test_the_host_wiring_path_actually_reaches_gate_2` in `tests/test_gate2.py` is the reachability pattern.
- `python -m rig.gate2_loop` drives six scenarios with no model and no network.
