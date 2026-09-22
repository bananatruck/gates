---
name: gate1-execution
description: Wire Gate 1 (execution validity) into a research scaffold's experiment phase, so only numbers a run recorded can be cited. Use when installing Gate 1, or when a scaffold decides an experiment worked by searching stdout or asking a model for a score.
---

# Gate 1: execution validity

Gate 1 decides whether the experiment ran, and which numbers it produced.
It opens at `GATES_LEVEL=1`, and both later gates read what it hands on.
Run `install-gates` first: its steps 1 and 2 check the scaffold can be gated and build the adapter this skill uses.

**Needs:** a subprocess the host controls, and experiment code that calls `record_result(key, value, unit=...)`.

**Hands on:** a `GatedExecution` whose `report` builds the registry of recorded values (`build_registry`), and whose `evidence_bundle` is what the writer reads in place of raw stdout.

## Declare

| Declaration | Feeds | Why the host must say |
|---|---|---|
| `expected_keys` | `results.expected_keys_present` | only the host knows what the task asked for |
| `task_ref` | the first link of each value's provenance chain | only the host holds the plan the run implements |

With no `expected_keys`, that check has no input and emits nothing.

## Wire it

1. **Prompt.** Add `MLE_GATE_INSTRUCTIONS` from `gates/pipeline.py` to the experiment agent's notes. It tells the agent to record values rather than print them.
2. **Context.** Build it with your adapter's `make_context(...)`, before the solver, so the solver can consult it on every attempt.
3. **Execution.** Replace the host's execute-and-judge call with your adapter's `gated_execute(code, context)`. Hand `feedback` back to the agent on a rejection.
4. **Turns.** Call `context.close_turn(passed)` once per agent turn, not per execution. Automated repair must not eat the agent's budget.
5. **Reward model.** Score only what passed: it ranks, it does not admit. Log the gate's verdict beside the reward with `record_divergence`.
6. **Final run.** Re-run the winning code under the gate, so the figures and the recorded values come from the same verified run.
7. **Spent budget.** `context.check_can_continue()` raises `GateFailure` when nothing ever passed. A run that never produced a valid experiment must not produce a paper.

Done when the host at `GATES_LEVEL=1` reaches a Gate 1 verdict on its own experiment, the writer receives `evidence_bundle` rather than stdout, and a run Gate 1 rejected never reaches the writer.

## Prove it

- `python -m rig.gate1_loop` replays the reject-fix-accept cycle with no model and no network (`rig/gate1_loop.py`).
- A host test that the experiment phase calls `gated_execute` on the winning code, and that level 0 runs the host's original path instead.
