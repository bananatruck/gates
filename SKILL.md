---
name: install-gates
description: Add the G.A.T.E.S. validity layer to an existing autonomous research scaffold. Use when a scaffold's agents can report numbers that were never measured, or when asked to install, port, or wire up G.A.T.E.S., Gate 1, Gate 2 or Gate 3.
---

# Installing G.A.T.E.S. into a research scaffold

You are adding a validity layer to a scaffold you did not write.
The claim this project makes is that hallucinated results are an information-flow defect rather than a model tendency, and portability is how that claim is tested.
So this file is the claim made executable: if installing takes host-specific surgery, the claim is weaker than stated.

Budget roughly **8 to 12 agent turns** for a scaffold whose experiment phase already executes code and captures stdout.
Most of that is step 2.

## Before you start

Read these, in this order, and do not skip the second:

1. `README.md` for what Gate 1 measured, and `docs/PLAN.md` for the design.
2. `CLAUDE.md` §2, the structural invariants. Every one of them is load-bearing on a published claim. If a step below seems to require breaking one, you have misread the step.

Four rules govern everything you write:

- `gates/` is Python-stdlib-only and never imports the host or `rig/`. Host knowledge lives in your adapter, and nowhere else.
- A model call can never block, fail, or change a verdict. Only deterministic FAIL-severity checks decide.
- A check with no input emits **nothing**. Never a passing row. An absent check is honest; a green check that never ran is a lie the paper inherits.
- A gate never mutates what it checks. It reads the artifact and writes evidence beside it.

## Step 1: find out whether the scaffold can be gated at all

Gate 1 verifies that reported numbers came from a run.
That is impossible unless the experiment *declares* its values, so check for this first and stop if it is missing.

Look for the point where the scaffold executes agent-written code and decides whether it worked.
In most scaffolds that decision is a string search over truncated stdout, or a language model asked for a score.
That is the defect. Note the file and line.

The experiment code must be able to call an injected `record_result(key, value, unit=...)`.
`gates.harness` injects it into the child process, so you do not write it.
What you do need is a prompt change telling the agent to call it, and a list of keys the run is expected to produce.
Without that list, `results.expected_keys_present` has no input and emits nothing.

If the scaffold cannot execute code in a subprocess you control, stop and say so.
Gate 1 needs the process boundary; there is no in-process fallback and inventing one would forfeit the isolation claim.

## Step 2: write the adapter

One new file, `gates/adapters/<host>.py`.
This is the only file allowed to know the host exists.
Copy `gates/adapters/agentlab.py` and work through it; it is the reference and it is commented for exactly this purpose.

An adapter owes the host four things:

**A context builder per phase.** One per gate, not one shared, because each phase has its own budget and its own rejection counter.
`make_context()` for Gate 1, `make_review_context()` for Gate 2, `make_report_context()` for Gate 3.
Each returns a `GateContext` holding the config, a `Ledger`, and the phase name the host itself uses.

**A gated call per phase**, wrapping `run_gate1`, `run_gate2`, `run_gate3`.
Each returns the verdict plus the feedback string the agent reads next turn.

**A loop per phase that closes.** `review_loop` and `report_loop` in the reference adapter.
The loop calls the host's agent, submits the artifact, and hands a rejection back as feedback.
This is the half that makes a gate a gate rather than a filter: a rejection that does not tell the agent what to fix is a stall.

**The host's own declarations**, at wiring time.
`gates/` holds no defaults for these and must not learn any:

| Declaration | Feeds | Why the host must say |
|---|---|---|
| expected result keys | `results.expected_keys_present` | only the host knows what the task asked for |
| `plan_fields` | Gate 2 tier B | never parsed from plan prose; `gates/` never reads a plan |
| `ranges`, `relations` | Gate 2 tier A | a metric's name never implies its range |
| `sections` | `style.sections_present` | which sections a paper needs is the venue's standard |
| `retrieved` | `source.cited_papers_in_registry` | only the host knows what it fetched |
| `lookup` | `source.identifiers_resolve` | `gates/` never opens a socket |

The last two are worth care.
`retrieved` is a **callable**, read after each write, because a scaffold that searches while it writes has not finished retrieving when the first draft appears.
`lookup` may raise; that is how it says the question could not be asked, and Gate 3 turns a raise into an INFO row rather than a rejection.
Returning `None` means the paper does not exist, which is a very different answer, and collapsing the two lets a network outage launder a fabricated citation.

## Step 3: place the call sites

Three, one per phase. Each replaces a decision the host was making badly.

**Gate 1, at the end of the experiment phase.** Build the context before the solver so the solver can consult it per attempt, then re-run the winning code under the gate so the figures and the recorded values come from the same verified run.
Feed the reward model only what already passed; it ranks, it does not admit.
On a spent budget, raise. A run that never produced a valid experiment must not produce a paper.

**Gate 2, after the verified run.** Review the registry Gate 1 wrote.
On a spent budget, **proceed** with the unresolved discrepancies carried forward as declared limitations. A genuine novel result must not be blocked forever.

**Gate 3, in the report-writing phase.** `report_loop` takes the registry the writer cites and the limitations Gate 2 declared.
On a spent budget, **raise**. An unverifiable manuscript is not emitted.

That asymmetry between Gates 2 and 3 is deliberate and you should state it wherever you document the install, or it reads as an inconsistency.

The writer needs a prompt change too: `REPORT_GATE_INSTRUCTIONS` in the reference adapter tells it to emit `\result{key}` tokens instead of typing numbers, and to place `\limitations{}` where the paper discusses its limitations.
The renderer substitutes both. A number the model typed has no key, and a key nothing recorded does not render.

## Step 4: budget in agent turns, with the cost on screen

`gates/setup.py` owns the warning text and takes one integer per gate.
An empty answer accepts the default.
Defaults live in the code and are tuned for completion and accuracy rather than for cost.

Budget state lives in your adapter, never in a gate.
No gate compares its own `attempt` against `max_attempts`; if you find yourself adding that, you are putting policy in the wrong half.

## Step 5: prove it is wired, not just imported

An adapter that compiles proves nothing. Write these, and watch each fail before it passes:

- **A reachability test per gate**, that the host's own wiring path arrives at a verdict. `test_the_host_wiring_path_actually_reaches_gate_2` and `test_the_host_wiring_path_actually_reaches_gate_3` are the pattern.
- **A completeness guard**: every check id the gate can emit has an evidence renderer, and every FAIL id has a fix directive. A keyed lookup that misses hands the agent a rejection it cannot act on. Extend the fixture whenever you add a check, or the guard passes vacuously.
- **A model-free loop**, in `rig/`, proving the reject-fix-accept cycle closes without an API key. `rig/gate3_loop.py` drives ten scenarios and needs no network and no model.
- **A key-leak test**, that no gate writes or transmits your provider credential. `tests/test_key_leak.py` plays all three gates with a sentinel key and fails if any written file or model prompt contains it.

To prove a guard actually bites, break the code it guards, watch the test fail, restore it, and confirm `git diff` is empty.
A guard nobody has seen fail is a guard you are trusting on faith.

## Worked example: Agent Laboratory

The reference host, and the comparison baseline.
It is already wired (D42).
Call sites in `ai_lab_repo.py`, as built:

| Phase | Method | What it calls |
|---|---|---|
| running experiments | `running_experiments` | `make_context`, then `gated_execute` on the winning code, then `review_loop(..., first=final)` if that run passed. `reviser_from_mle_solver` is the `revise` callback. |
| report writing | `report_writing` | Refuses if Gate 1 left no citable registry. Then `report_loop` with `arxiv_lookup`, `writer_from_paper_solver`, and `retrieved_arxiv_ids(self.phd.lit_review, solver.section_related_work)`. |

The host's two callbacks as built (`1966e17`) predate the rules below: they return the reward-best entry, keep notes as a list, and hand the writer the pre-review code. Those are open host items, recorded in `progress.md` as F13-F15, not properties of this recipe.

For the writing phase the `write` callback wraps the host's solver.
The first call runs `solver.initial_solve()`, then the host's usual number of `solver.solve()` steps, and returns `"\n".join(solver.best_report[0][0])`.
Keeping the host's own effort matters for any gated-versus-ungated comparison: a first draft Gate 3 admits must not get fewer improvement steps than the ungated writer.
Each later call does three things:

1. **Voids the rejected draft's reward score**, setting it to `float("-inf")`. The solver keeps one best draft and replaces it only on a higher score, so without this a fix the reward model likes less is discarded and the rejected draft is resubmitted until the budget raises.
2. **Appends the feedback to `solver.notes` as text**, `f"{solver.notes}\n{feedback}"`. The host interpolates notes straight into its prompt, and a list renders as its repr: every `\result{}` gains a second backslash and the rejection collapses onto one line.
3. **Runs one `solver.solve()` and returns the draft it produced**, or `None` if the draft still scores `-inf`, since nothing new was written and resubmitting would spend a turn to hear the same rejection.

`tests/test_install_skill.py` drives this recipe through the real `report_loop` against a solver that ranks the way the host does.
`revise` for Gate 2 follows the same three rules against `solver.best_codes`, tracking the entry it last returned, since the experiment solver keeps two.
After the loop, hand the writer `ReviewOutcome.run.code` and `ReviewOutcome.run.evidence_bundle`, not the code the phase started with: once a revision is reviewed, the registry the writer cites came from that run.
Retrieval is `lambda: retrieved_arxiv_ids(self.phd.lit_review, solver.section_related_work)`, which reads the host's two formats: `lit_review` entries keyed `arxiv_id`, and per-section arXiv search results as text carrying `arXiv paper ID:` lines.

What this host cannot support, and why that is fine to report:

- **No bibliography.** It cites inline as `(arXiv 2308.11483v1)`, so `source.citations_parse` and `source.metadata_agrees` have no input and emit nothing. Do not synthesise one to make a check run.
- **Free-text plans.** Nothing extracts `plan_fields`, so Gate 2 tier B only activates if a host declares them at wiring time.
- **No DOIs.** The canonical identifier is the arXiv id, compared version-stripped.

## When you are done

Report what you wired, what emitted nothing and why, and what you could not check.
The last part is the important one.
A gate that reports only what it verified, and names what it did not, is the whole product.
Overclaiming in that report is the same defect the gates exist to catch, committed by us.
