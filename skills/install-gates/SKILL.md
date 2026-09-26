---
name: install-gates
description: Add the G.A.T.E.S. validity layer to an existing autonomous research scaffold. Use when a scaffold's agents can report numbers that were never measured, or when asked to install, port, or wire up G.A.T.E.S., Gate 1, Gate 2 or Gate 3.
---

# Installing G.A.T.E.S. into a research scaffold

You are adding a validity layer to a scaffold you did not write.
The claim this project makes is that hallucinated results are an information-flow defect rather than a model tendency, and portability is how that claim is tested.
So these skills are the claim made executable: if installing takes host-specific surgery, the claim is weaker than stated.

Budget roughly **8 to 12 agent turns** for a scaffold whose experiment phase already executes code and captures stdout.
Paths below are relative to the root of the gates repository.

## Before you start

Read these, in this order, and do not skip the second:

1. `README.md` for what Gate 1 measured, and `docs/PLAN.md` for the design.
2. `CLAUDE.md` §2, the structural invariants. Every one of them is load-bearing on a published claim. If a step below seems to require breaking one, you have misread the step.

Four rules govern everything you write:

- `gates/` is Python-stdlib-only and never imports the host or `rig/`. Host knowledge lives in your adapter, and nowhere else.
- A model call can never block, fail, or change a verdict. Only deterministic FAIL-severity checks decide.
- A check with no input emits **nothing**. Never a passing row. An absent check is honest; a green check that never ran is a lie the paper inherits.
- A gate never mutates what it checks. It reads the artifact and writes evidence beside it.

## The level switch

One environment variable picks which gates run, and the levels are cumulative because each gate reads what the one before it produced.

| `GATES_LEVEL` | Gates open | Meaning |
|---|---|---|
| `0` | none | the host exactly as shipped, the control arm |
| `1` | Gate 1 | Gates 2 and 3 closed |
| `2` | Gates 1 and 2 | Gate 3 closed |
| `3` | all three | the default when nothing is set |

Your host code asks `gate_level()` before each phase's call site.
Every gate entry point in `gates/pipeline.py` also calls `require_gate()`, so a closed gate raises before it spends an agent turn.
`GATES_GATE1=off` is the older switch and still means level 0; setting both raises.

## Step 1: find out whether the scaffold can be gated at all

Gate 1 verifies that reported numbers came from a run.
That is impossible unless the experiment *declares* its values, so check for this first and stop if it is missing.

Look for the point where the scaffold executes agent-written code and decides whether it worked.
In most scaffolds that decision is a string search over truncated stdout, or a language model asked for a score.
That is the defect. Note the file and line.

The experiment code must be able to call an injected `record_result(key, value, unit=...)`.
`gates.harness` injects it into the child process, so you do not write it.

If the scaffold cannot execute code in a subprocess you control, stop and say so.
Gate 1 needs the process boundary; there is no in-process fallback and inventing one would forfeit the isolation claim.

## Step 2: write the adapter

One new file, `gates/adapters/<host>.py`, the only file allowed to know the host exists.
It holds host knowledge and nothing else, because the loops every host shares live in `gates/pipeline.py` and you import them.
Read `gates/adapters/agentlab.py` as the example of the shape.

An adapter owes the host five things:

- **A context builder per gate**: `make_context()`, `make_review_context()`, `make_report_context()`. Each returns a `GateContext` with its own budget and rejection counter, a `Ledger`, and the phase name the host itself uses.
- **The model call** each gate's advisory layer uses, as a two-argument function. Import the host's client lazily, inside the function, or `pip install gates` breaks for everyone else.
- **The level-0 path**: a `gated_execute` that runs the host's original execution when `gate_level()` is 0 and `gates.pipeline.gated_execute` otherwise.
- **Retrieval**: the ids of the papers the host fetched, read from the host's own formats.
- **The host's declarations**, each described in the skill for the gate that reads it. `gates/` holds no defaults for these and must not learn any.

## Step 3: install the gates in order

Each gate is its own skill, and each needs the one before it:

1. `gate1-execution`: the experiment phase. Hands on a registry of recorded values.
2. `gate2-coherence`: reviews that registry against the plan's declarations. Hands on the reviewed run and its declared limitations.
3. `gate3-report`: the writing phase. Renders every number from the registry and admits the manuscript.

Stop at the level the host needs, and finish each skill's proof before starting the next.

The exhaustion policies differ on purpose: a spent Gate 2 budget **proceeds** with its limitations declared, and a spent Gate 1 or Gate 3 budget **raises**.
State that asymmetry wherever you document the install, or it reads as an inconsistency.

## Step 4: budget in agent turns, with the cost on screen

`gates/setup.py` owns the warning text and takes one integer per gate.
An empty answer accepts the default.
It prints the budgets as JSON, and your host reads them back with `parse_budgets()` and passes each as its context's `max_attempts`; Agent Laboratory takes them as `--gate-budgets`.
Defaults live in the code and are tuned for completion and accuracy rather than for cost.

Budget state lives in your adapter, never in a gate.
No gate compares its own `attempt` against `max_attempts`; if you find yourself adding that, you are putting policy in the wrong half.

## Step 5: prove the wiring runs

Each gate skill names its own reachability test and model-free loop.
Two proofs cover the whole install:

- **A level test per arm**: at each `GATES_LEVEL` the host runs, the closed gates never run and the open ones do.
- **A key-leak test**, that no gate writes or transmits your provider credential. `tests/test_key_leak.py` plays all three gates with a sentinel key and fails if any written file or model prompt contains it.

To prove a guard bites, break the code it guards, watch the test fail, restore it, and confirm `git diff` is empty.
A guard nobody has seen fail is a guard you are trusting on faith.

## Worked example

Read `agentlab-example.md` in this folder when the host is Agent Laboratory, or when you want the reference host's call sites and callbacks as built.

## When you are done

Report what you wired, what emitted nothing and why, and what you could not check.
The last part is the important one.
A gate that reports only what it verified, and names what it did not, is the whole product.
Overclaiming in that report is the same defect the gates exist to catch, committed by us.
