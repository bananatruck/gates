# Gate 1 — what is left, and how to finish it

**Read this with [`PLAN.md`](PLAN.md) §7 open.** Gate 1's implementation is complete: 16 checks
live, 206 tests green, wired into Agent-Researcher, zero model calls in the verdict. Everything
below is *measurement* — the numbers the paper needs, which no amount of further coding produces.

## The LLM layer

Built, and required rather than optional — the feedback report is the loop's return path to the
ML engineer, and no template writes `PLAN.md` §3.4's example. It does two jobs, both after
`decide()` has fixed the verdict:

| Job | Module | Severity | Measured against |
|---|---|---|---|
| Read the log lines the pattern set cannot reach | `gates/llm_scan.py` | WARN, by construction | `tests/fixtures/log_corpus.jsonl` |
| Write the REQUIRED FIXES the engineer reads | `gates/llm_report.py` | not a check at all | `rig/tuning.py` (unrun) |

The deterministic baseline is now a number rather than an adjective:

```
precision  1.000  (95% CI 0.824–1.000)   18 of 18 flagged were real
recall     0.529  (95% CI 0.367–0.685)   18 of 34 signals found
```

Precision is the floor and the corpus enforces it. Recall is what the model layer exists to move,
and **the model's own rates are not yet measured** — that needs an API key, and is T6 below.

Status of the three source documents' requirements is unchanged and traced in
[`GATE1_REQUIREMENTS.md`](GATE1_REQUIREMENTS.md): V1–V6 met, D1–D7 met, D8 instrumented, D9
pending measurement, P1–P4 and P6–P10 met, P5 partial by design (it is Gate 2's), P11–P12 out of
scope.

---

## 0. Two corrections to the plan, verified

| Claimed | Actual |
|---|---|
| `PLAN.md` step 4: "Fix `run_experiments.py --yaml-location`" | **Already fixed.** `CLI_OPTIONS` no longer contains it; `build_command` emits no flag `ai_lab_repo.parse_arguments` rejects. Checked by constructing the command from `sgc_gemini_3_5_flash.yaml` and differencing it against the parser's flag set — zero unrecognized. |
| Deck slide 6: "n = 1 (5 runs died on `--yaml-location`)" | The *cause* is fixed; the *consequence* is not. n is still 1, because the runs have not been repeated. |

A third defect of the same shape was found while fixing this and is now closed — five YAML keys
that reached nothing, described in T1 below.

One thing still blocks the re-run and is a **decision, not a defect**: the configs the dead runs
used no longer exist. `results/` references `sgc_gemini_3_1_flash_lite.yaml` and
`sgc_gemini_3_5_flash_lite.yaml`; `experiment_configs/` holds only the two non-lite files. The
model set for the re-run has to be chosen rather than recovered. A wrong model id now fails in
the preflight (`run_experiment` checks the name against the live model list before starting), so
this costs seconds rather than a run.

---

## 1. The tasks

Ordered by what unblocks what. T1 and T2 are the critical path to a defensible n.

### T1 — Make the runner's configuration honest — **DONE**

All five dropped keys turned out to be inherited from upstream Agent Laboratory, which reads its
YAML directly; this fork rewrote `ai_lab_repo.py` around argparse and never carried them over.
None appeared anywhere in the fork's Python. Resolved one of two ways each:

| Key | Resolution |
|---|---|
| `lit-review-backend` | **Wired through.** New `--lit-review-backend` flag; `agent_models["literature review"]` uses it, falling back to `--llm-backend` when absent. It is the one key that changes what a run *is*. |
| `num-papers-to-write`, `parallel-labs`, `lab-index`, `except-if-fail` | **Deleted from the configs.** All belong to upstream's multi-lab / AgentRxiv path, which this fork does not implement. |

`build_command` now refuses an unmapped key instead of dropping it:

```
config keys reach nothing: papersolver-maxsteps. Add each to CLI_OPTIONS and to
ai_lab_repo.py's parser, or delete it from the config. A run must be described by
the file that configured it.
```

Held by `tests/test_run_experiments_config.py` in the host repo — every config's keys must reach
the parser, every mapped flag must exist, an unmapped key must raise, and the built command must
contain no argument the parser rejects. That last test is the one that would have caught
`--yaml-location` before it cost five runs. Scaffold suite: 49 tests, up from 41.

### T2 — Re-run the archive for a real n *(host repo, the expensive one)*

The audit baseline is n=1. Nothing about a rate is defensible until this runs.

| Arm | Config | Purpose |
|---|---|---|
| **gated** | Gate 1 live, `require_metrics=True` | The layer's behaviour on real agent output |
| **ungated** | `MAX_LEN=1000`, no gate — upstream's channel | The baseline fabrication rate |

Five runs per arm per model is the minimum that supports a proportion with an interval worth
printing; the archived run took 5,836s, so budget accordingly and run them unattended. Two
one-line changes in `run_experiments.py` are needed first, and neither is currently set for a
measurement campaign:

```python
RUN_INDICES = (1,)              # → (1, 2, 3, 4, 5)
STOP_MODEL_AFTER_FAILURE = True # → False: a failed run is data, not a reason to stop
```

The second matters more than it looks. Leaving it `True` means the arm that fails most —
precisely the one the paper is about — contributes the fewest runs, and the sample silently
selects for success.

Two things to record per run that the current wiring does not yet force:

- **The ungated arm needs a writing agent.** Fabrication rate is a property of the paper, not of
  the execution, so the ungated arm must run all the way to a manuscript.
- **Label the fabricated numbers by hand.** D9's "numeric fabrication rate" needs a human
  comparing each reported number against the run's actual capture. There is no way around this
  and it should be budgeted as annotation time, not as compute.

**Done when:** `divergence.jsonl` holds ≥ 10 gated attempts across ≥ 5 runs, and the gate-vs-
reward table in the paper is generated from it rather than from the single archived run.

### T3 — Close the channel-fidelity experiment *(mixed)*

D8, deck slides 10 and 15. Two arms, and they have different costs now:

| Arm | Question | Cost |
|---|---|---|
| **detector** | At what `MAX_LEN` does upstream's crash marker survive? | **None — already measurable.** `rig/reward.channel_sweep` answers it offline, per capture, and `tests/test_loop.py` holds the result. |
| **writer** | Does fabrication rate fall as the channel widens? | Real runs at `MAX_LEN ∈ {1000, 4000, 16000, ∞}`, with `Gate1Config(require_metrics=False)` so the degraded arm is observed rather than blocked. |

The detector arm is worth reporting on its own: for the archived crash shape the marker lands at
character 3,442, so a 1,000-character channel could not have seen it and a 4,000-character one
could. That is a mechanism, stated exactly, with no model involved.

**Done when:** the writer arm has run at ≥ 3 widths and the fabrication rate at each is in the
paper with its interval.

### T4 — Second adapter *(gates repo, ~1 day)*

The deck says "1 host only" and the paper argues portability. One more adapter — AI-Scientist-v2
is the closest target — converts that argument into a demonstration. The contract is four things
(`PLAN.md` §6.1); if a second host needs a fifth, that is itself a finding about the contract.

**Done when:** `gates/adapters/<host>.py` exists, `gates/` still imports nothing from any host,
and the new host's integration tests pass.

### T5 — Settle the one open design decision — **RESOLVED: `GateFailure` stands**

Deck slide 6, row 3. As described: budget exhausted → pass the report to Gate 2. As built:
budget exhausted with nothing passing → `GateFailure`, run exits non-zero, no paper.

Resolved in favour of what is built, because the deck contradicts itself and only one branch is
survivable. Slide 12 and `PLAN.md` §5.2 both claim:

> Silent failure scored as success — **eliminated by construction** (Gate 1) — a crashed run
> cannot reach writing.

"Eliminated by construction" is true only if a rejected run is terminal. Let the exhausted budget
hand its report to Gate 2 and the claim demotes to *reduced and measured*, which costs one of the
three MLR-Bench classes the paper eliminates outright — the strongest thing Gate 1 has. The
weaker branch buys nothing in exchange: Gate 2 compares measured numbers against the literature,
and a run that never produced a measurement gives it nothing to compare.

Note precisely what is and is not terminal, because the distinction does the work: `GateFailure`
fires only when the budget is spent **and no attempt ever passed**. If any attempt passed, the
phase falls back to it and continues. The strict reading rejects unexecuted experiments, not
imperfect ones.

**Action:** slide 6 row 3 should read `GateFailure — no paper` in both columns, and drop out of
the "three things to settle" list. `PLAN.md` §3 and the code already agree and need no change.
Overrule this if you want the softer terminal state — it is a claim about what the layer is for,
and it is yours to make — but the paper's eliminated-by-construction table has to change with it.

---

## 2. The feedback loop — how to test any of this now

`rig/` drives the real gate through complete engineer turns with a scripted engineer in place
of a model. Every code path is the shipped one — `gated_execute`, `run_gate1`, `render_feedback`,
`Ledger`. Only the source of the code is fake, which is what makes it free.

```bash
python -m rig.gate1_loop                     # every scenario, full transcript
python -m rig.gate1_loop archived-run        # one
python -m rig.gate1_loop --list
python -m rig.gate1_loop --quiet             # summary table only
python -m rig.gate1_loop --json              # for CI
python -m rig.gate1_loop --workdir ./run     # keep the artifacts
```

Exit status is non-zero if any scenario departs from what it documents, so this is a test as much
as a demonstration. It is also held by pytest — `tests/test_loop.py`, 18 tests — so a regression
in the loop fails the same suite as a regression in a check.

### The scenarios

| Scenario | What it demonstrates | Ends in |
|---|---|---|
| `archived-run` | The audited run, replayed: unbound name → crash behind heavy output → runs clean and records nothing. Three turns, three different checks. | `GateFailure` — no paper |
| `recovers` | Rejection → the agent invents numbers instead → a real measurement. The intended path. | PASS on turn 3, evidence bundle emitted |
| `inner-repair` | Three executions inside **one** engineer turn. The budget charges one. | PASS, 0 rewrites spent |
| `warn-tier` | Caught exception, NaN warning, CPU fallback, no seed, exact-zero metric. Four warnings, none blocking. | PASS, warnings in the bundle |
| `namespace-leak` | A name bound by a passing run is gone by the next one. | Turn 2 rejected at runtime |

Each scenario declares the checks each execution must fail or warn on, the number of engineer
turns it should consume, and its terminal state; `check_expectations` compares the run against
that declaration. A scenario cannot drift from its own description without the rig going red.

### What the rig measures, and what it refuses to

The interesting column is **upstream blind**: attempts Gate 1 rejected that the host scaffold's
own failure detector would have accepted. That detector is a substring search over a
1,000-character slice, so it can be reproduced exactly, and it is — including the honest awkward
case, where an attempt the static tier rejected *before execution* has to actually be run to know
what upstream would have seen (`rig/reward.shadow_execute`, flagged in the ledger as a
reconstruction).

```
  SCENARIO         TURNS  EXEC  OUTCOME       UPSTREAM BLIND  RESULT
  archived-run         3     3  gate_failure  2/3 turns       ok
  recovers             3     3  pass          1/3 turns       ok
  inner-repair         1     3  pass          —               ok
  warn-tier            1     1  pass          —               ok
  namespace-leak       2     2  pass          —               ok

  Gate 1 rejected 6 engineer turn(s); upstream's 1,000-character detector
  would have accepted 3 of them.
```

What the rig will **not** do is invent a reward score. `get_score` is an LLM at temperature 0.6
and cannot be reconstructed, so `reward_score` stays `null` in the ledger unless a real model is
supplied:

```python
run_loop(SCENARIOS["archived-run"], workdir=..., reward_fn=my_model)
```

`reward_fn` receives the 1,000-character view upstream would have had — not the full capture —
so what lands in the ledger is the counterfactual the paper needs rather than a rescored run.
This is the seam where T2's real model plugs in.

### Plugging in a real engineer

The loop does not care where code comes from. Implement one method:

```python
class ModelEngineer:
    def turn(self, feedback: str | None, turn_index: int) -> list[Step] | None:
        if feedback is None:
            return [Step("initial", self.model(self.task))]
        return [Step(f"rewrite {turn_index}", self.model(self.task, feedback))]
```

Pass it as `run_loop(..., engineer=ModelEngineer())`. That converts the rig from a fixture player
into a live loop with a real agent, on the same artifacts, ledger and expectations — which is the
cheapest available path to T2's numbers on a single machine before committing to full runs.

---

## 3. Summary

| # | Task | Where | State |
|---|---|---|---|
| T1 | Configuration keys must not be dropped silently | host repo | **done** |
| T5 | Settle: budget exhausted → `GateFailure` or → Gate 2 | decision | **resolved — `GateFailure`** |
| — | LLM layer: log scan, feedback generation, grounding, cost ceiling | gates repo | **done** |
| T2 | Re-run the archive, both arms, for a real n | host repo | **open** — needs API budget + hand annotation |
| T3 | Channel-fidelity: writer arm | host repo | **open** — needs T2's runs |
| T4 | Second adapter | gates repo | **open** — ~1 day, no API cost |
| T6 | Measure the model tier: scan precision/recall, and feedback convergence | gates repo | **open** — harness built, needs an API key |

**Gate 1 is complete as an implementation.** Every item that can be closed without spending API
budget or annotation time is closed. What remains is T2, T3 and T6, which are measurement — they
need real runs against a real model and a human labelling the fabricated numbers — and T4, which
is a portability demonstration rather than a Gate 1 requirement.

T6 is the cheapest of the three and unblocks the model layer's claims:

```python
from rig.corpus import combined_scanner, load_corpus, score, render
print(render("deterministic + model", score(combined_scanner(my_model_fn))))

from rig.tuning import compare_arms
print(compare_arms(my_model_fn, seeds=5))
```

The first is 68 model calls and produces the precision/recall pair the layer is claimed on. The
second runs three tasks twice under two feedback regimes and produces turns-to-pass. Neither
needs a training run.

To launch T2 when the budget is there:

```bash
cd AgentLaboratory-Gemini
python run_experiments.py            # configs are honest now; a bad key fails in a second
```

Nothing in `gates/` is waiting on any of it.
