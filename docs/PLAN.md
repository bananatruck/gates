# G.A.T.E.S., a portable validity layer for autonomous research agents

**Status:** Gates 1-3 built and tested; Gate 1 measured on a live scaffold; the benchmark evaluation is planned in [`paper/PLAN.md`](../paper/PLAN.md).
**Scope:** three deterministic-first gates inserted into an existing autonomous-research scaffold.
**Host scaffold (testbed):** Agent-Researcher, a fork of Agent Laboratory.
**Portability target:** AI-Scientist-v2, Agent Laboratory upstream, and systems like ScientistOne.

This is the design.
`paper/PLAN.md` is the evaluation, and `progress.md` is the decision log and the current state.

---

## 0. The claim

> A measurable share of hallucinated experimental results in autonomous research agents comes from lossy or absent evidence channels inside the scaffold, not from model dishonesty.
> Three deterministic gates at the channel boundaries eliminate three of the four MLR-Bench hallucination classes by construction and reduce the fourth to a measured rate.

The gates are the contribution.
The auto-researcher is the testbed.

§5.2 revises the second sentence against what was built: two classes are eliminated by construction, and two are detected and measured.

### Why this is not already solved

| System | What it does | What it leaves open |
|---|---|---|
| AutoResearchClaw (arXiv 2605.20025) | Builds a verified value registry during execution | Passes zero-valued results, which are real measurements of degenerate science |
| SAGE (arXiv 2606.31478) | Grounded reporting that redacts hallucinated numbers | Gates the output; the writer still works without the evidence |
| ScientistOne (arXiv 2605.26340) | A chain of evidence per claim | Nothing audits whether the evidence supports the claim |

All three treat fabrication as a model tendency and catch it at the output.
G.A.T.E.S. treats it as an information-flow defect and closes the channel at three points.
Gate 1 sits upstream of every published approach.
It fires before the reward model, before interpretation, and before writing.

---

## 1. Baseline: what the host scaffold does today

We verified every number below in the host repository.

### 1.1 The evidence bottleneck

`tools.py:383`

```python
def execute_code(code_str, timeout=60, MAX_LEN=1000):
    ...
    return output_capture.getvalue()[:MAX_LEN]
```

That 1,000-character prefix is the whole experimental record.
It flows here:

```
execute_code() ──► args[1] ──► best_codes[i][2] ──► exp_results
                                                       │
   ai_lab_repo.py:347 ──────────────────────────────────┘
                                                       ▼
   papersolver.py:553  "After running this code, the following results were observed: {exp_results}"
```

The host asks the writing agent for a results table and gives it section headers with no values.

### 1.2 The bottleneck also blinds the error detector

This finding changes the diagnosis.
`execute_code` writes the error marker to the capture buffer after the program's own output:

```python
except Exception as e:
    output_capture.write(f"[CODE EXECUTION ERROR]: {str(e)}\n")
    traceback.print_exc(file=output_capture)
```

Once an experiment prints more than 1,000 characters, and every real experiment does, the marker falls off the end of the slice.
The solver's only crash test is a substring search for that marker (`mlesolver.py:91`):

```python
if "[CODE EXECUTION ERROR]" in code_ret: return False, (None, code_ret,)
```

So the crash is invisible.
From `results/gemini_3_5_flash_run_1/`, log lines 700-745:

```
NameError: name 'hidden_dim' is not defined
$$$$ CODE REPLACE (success)
$$$ Score: 0.98
Running experiments completed, reward function score: 1.0
```

One line of code causes both the silent failure and the fabrication.
They are one bug, not two.

### 1.3 Execution has no isolation

`tools.py:402` runs `exec_globals = globals()`.

Every experiment executes in `tools.py`'s own module namespace, and nothing clears it between attempts.
That has four consequences.

- A name bound by attempt *N* is still bound in attempt *N+1*, so an experiment can pass because an earlier one defined a variable. This is the "are the variables real" question.
- Generated code can rebind `tools.py` internals such as `sys`, `os`, `traceback` and `execute_code`.
- Nothing redirects `stderr`, so anything the experiment prints there is lost.
- Execution runs in a `ThreadPoolExecutor`. `future.result(timeout=...)` returns but cannot kill the thread, so a runaway experiment keeps burning CPU for the rest of the run, with `sys.stdout` swapped for the whole process.

### 1.4 The only quality gate is a model grading a model

`mlesolver.py:151`: `get_score()` asks an LLM at `temp=0.6` for a float in `[0,1]`.
The host tests validity by whether `float()` parses the reply, not by whether the run worked:

```python
score, cmd_str, is_valid = get_score(...)
if is_valid:
    failed = False
    break
```

`is_valid` means the model returned a number.
The crashed run above scored 0.95, then 0.98, then 1.0.

### 1.5 Archive state

Seven runs are archived under `results/`, and one completed.
Five died at `ai_lab_repo.py: error: unrecognized arguments: --yaml-location`, and one on a `gemini-2.5-flash-lite` 404.
The audit baseline is n=1, not n=7.
§7 records what happened to the argument bug and the re-run.

---

## 2. Architecture

```
                    ┌──────────────────────────────────────────────┐
  user prompt ─────►│  review → plan → executable task list        │
                    └──────────────────────┬───────────────────────┘
                                           ▼
                    ╔══════════════ EXECUTION LABORATORY ══════════════╗
                    ║                                                  ║
                    ║   ML engineer ──► writes / edits code            ║
                    ║        ▲                    │                    ║
                    ║        │                    ▼                    ║
                    ║        │            sandboxed execution          ║
                    ║        │                    │                    ║
                    ║        │                    ▼                    ║
                    ║        │        ┏━━━━━━━━━━━━━━━━━━━━━━┓         ║
                    ║        └────────┨  GATE 1              ┃         ║
                    ║   feedback      ┃  execution validity  ┃         ║
                    ║   report        ┗━━━━━━━━━━┳━━━━━━━━━━━┛         ║
                    ║   (max 3)                  │ PASS               ║
                    ║                            ▼                    ║
                    ║              get_score()  - tie-break only      ║
                    ║              divergence.jsonl ◄── evidence      ║
                    ║                            │                    ║
                    ║        ┌───────────────────┤                    ║
                    ║        │                   ▼                    ║
                    ║        │       ┏━━━━━━━━━━━━━━━━━━━━━━┓         ║
                    ║        └───────┨  GATE 2              ┃         ║
                    ║   feedback     ┃  source ↔ result     ┃         ║
                    ║   report       ┃  coherence           ┃         ║
                    ║   (max 2)      ┗━━━━━━━━━━┳━━━━━━━━━━━┛         ║
                    ╚════════════════════════════┿═════════════════════╝
                                                 │ PASS
                                                 ▼
                                    finalized evidence bundle
                                    (results.json + registry + full stdout)
                                                 │
                                                 ▼
                              report generation ◄──┐
                                     │             │ feedback
                                     ▼             │ report
                          ┏━━━━━━━━━━━━━━━━━━━━━┓  │
                          ┃  GATE 3             ┃──┘
                          ┃  report validity    ┃
                          ┗━━━━━━━━━┳━━━━━━━━━━━┛
                                    │ PASS
                                    ▼
                            finalized report
```

`GATES_LEVEL` opens these gates one at a time, cumulatively: 0 is the host as shipped, and 3 is all three (D58).

### 2.1 Design invariants

These hold for all three gates, and they make the layer portable.

1. **Deterministic first.** A gate fails on a mechanically checkable fact wherever one exists. A model is consulted only where none exists, which means Gate 1's log recall and feedback prose, Gate 2's semantic coherence, and Gate 3's prose entailment. Its output is reported as a rate with a confidence interval, never claimed as a guarantee. The rule constrains where a model sits, not whether one is present. Every verdict is deterministic, and Gate 1's LLM layer is safe because it sits outside the verdict (§3.4).
2. **The gate is the authority.** No model opinion can admit an artifact the gate rejected.
3. **Every gate emits two artifacts.** One is a machine-readable verdict, `gate<N>_report.json`. The other is a short feedback report for the agent the gate loops back to.
4. **Bounded loops.** Every gate has a retry budget with a defined end state, so no gate can spin forever.
5. **Zero scaffold imports.** `gates/` imports nothing from the host. All host knowledge lives in `adapters/`.
6. **Evidence is append-only.** Gates never edit agent output. They accept, reject, or annotate.

### 2.2 Severity model

| Severity | Effect |
|---|---|
| `FAIL` | The verdict is FAIL. The gate rejects the artifact and the feedback loop fires. |
| `WARN` | Recorded in the report and shown to the agent. Does not block. |
| `INFO` | Provenance and evidence only. |

The verdict is `FAIL` if any `FAIL` check failed, else `PASS`.

---

## 3. Gate 1, execution validity

> **Question:** did this code run to completion, and did this run produce the numbers it reports, rather than inheriting, hardcoding, or inventing them?

**Position:** between code execution and the reward model, inside the MLE solver loop.
**Loops back to:** the ML engineer agent.
**Retry budget:** 3 rewrites, chosen at setup.
**On exhaustion:** fall back to the latest attempt that passed Gate 1. If none ever passed, raise `GateFailure` and exit non-zero. No paper comes from an experiment that never ran.

### 3.1 Requirements

| # | Requirement | Why |
|---|---|---|
| R1.1 | Experiment code runs in a fresh OS process with an empty `__main__` namespace | Ends the `exec_globals = globals()` leak and makes "real variables" checkable |
| R1.2 | `stdout` and `stderr` are captured separately and in full, to files, never cut | Retires `MAX_LEN=1000` and restores the evidence channel |
| R1.3 | Failure detection reads the process exit code, not a substring of stdout | Removes the coupling that let truncation hide crashes |
| R1.4 | A timeout kills the whole process group | A runaway experiment must not outlive its own timeout |
| R1.5 | Experiments declare results through a typed contract, `results.json`, not by printing | Downstream code stops parsing prose |
| R1.6 | Every declared metric records its call site: file, line and source text | Enables the anti-literal check and Gate 3's chain of evidence |
| R1.7 | A metric whose value is a source literal fails the gate | A typed number is not a measurement |
| R1.8 | Unbound names are found statically, before execution | A 45-minute run should not be spent finding a `NameError` |
| R1.9 | The verdict is computed with no LLM in the path | The runtime already knows whether the run succeeded |
| R1.10 | Every attempt is logged to `divergence.jsonl` with both the gate verdict and the reward score | The disagreement between them is a headline result |
| R1.11 | Both captured streams are scanned for errors the run continued past, reported and never blocking | The spoken spec asked for "log files without errors that don't seem really code-breaking". A clean exit code cannot see a caught exception, a divide-by-zero NaN, or a CUDA fallback |
| R1.12 | Every recorded value is bound to one execution by a trace id, and the process that ran the source hashes it | Deck slide 11: "typed and hashed to the run that produced it". Without the hash, the binding is an assumption |
| R1.13 | Each value carries its provenance chain link by link, with unresolved links reported, not omitted | ARGUS Sub-Topic C: a value that merely matches a log is a worse signal than a numeric discrepancy, so the two must be told apart |
| R1.14 | Every `record_result` call is kept, not only the last | ARGUS Sub-Topic B: a best-epoch number reported as a final-epoch one is invisible if earlier calls are discarded |
| R1.15 | A run that declares no seed is reported as unreproducible | Re-running the code to check a claim is impossible without one |

[`GATE1_REQUIREMENTS.md`](GATE1_REQUIREMENTS.md) traces each source requirement to the check and test that meet it, including the one met only in part (P5, undisclosed replicate drops).

### 3.2 Checks

Static checks run before execution, so a rejection costs no compute:

| ID | Check | Severity |
|---|---|---|
| `static.syntax_valid` | The source compiles | FAIL |
| `static.no_unbound_names` | Scope analysis finds no name loaded but never bound and not a builtin or import | FAIL |
| `static.no_banned_calls` | No `exit()`, `sys.exit()` or `os._exit()`, which forge a clean exit code | FAIL |

Runtime:

| ID | Check | Severity |
|---|---|---|
| `exec.exit_code_zero` | The subprocess exited 0 | FAIL |
| `exec.no_uncaught_exception` | The harness recorded no escaping exception | FAIL |
| `exec.completed_within_budget` | The timeout did not kill it | FAIL |
| `env.clean_namespace` | The initial global namespace held only harness-provided names | FAIL |
| `env.code_identity` | The source the child hashed matches the source the parent wrote | FAIL |
| `exec.no_swallowed_traceback` | No `Traceback` in either stream from a run that still exited 0 | WARN |
| `logs.no_error_signals` | No numerical-integrity warning, device failure, non-convergence, printed exception or `ERROR`-level log | WARN |
| `env.seed_recorded` | The experiment declared a seed | WARN |

Results contract:

| ID | Check | Severity |
|---|---|---|
| `results.contract_present` | `results.json` exists and parses | FAIL |
| `results.expected_keys_present` | Every key the plan declared is present | FAIL |
| `results.values_computed` | No metric value is an `ast.Constant` at its record call site | FAIL |
| `results.values_finite` | No `NaN` and no infinity | FAIL |
| `results.single_observation` | No key was recorded many times with changing values | WARN |
| `results.non_degenerate` | Flags exact `0.0`, exact `1.0`, and exact chance level when the class count is known | WARN |

`results.non_degenerate` answers AutoResearchClaw's stated limitation, that a registry alone passes real zeros.
Gate 1 reports them and does not accept them silently.

`logs.no_error_signals` is the WARN tier the spoken spec asked for: errors in the logs that did not break the run.
Its patterns favour precision, because a false positive costs the agent a rewrite for nothing.
It flags `ValueError:` and ignores `Mean Squared Error:`, and a test holds each direction.
Its recall has no bound, and the paper says so.

`results.single_observation` exists because the harness keeps every `record_result` call.
The gate cannot know which of an epoch loop's values the paper means, so it reports the span and makes the report say.

Provenance, recorded and never gating:

| ID | Recorded | Severity |
|---|---|---|
| `output.untruncated` | stdout and stderr byte counts, and that nothing cut them | INFO |
| `env.provenance` | Python version, platform, seed, device, key library versions, code SHA-256 | INFO |
| `env.parent_proc_guard` | whether the experiment could read the gate's own process: `active`, or why not (`bypassable` under CAP_SYS_PTRACE, `failed`, `unsupported`) | INFO |

### 3.3 The results contract

The harness injects the results API into the experiment's namespace, so there is no install, no import path, and nothing for the agent to get wrong:

```python
record_result("exp1.K2.test_acc", test_acc, unit="ratio")
record_result("exp2.sgc.wallclock_s", total_sgc_time, unit="seconds")
```

`record_result` captures the caller's frame: file name, line number, and the source text of the call.
Gate 1 then re-parses the source, finds the `Call` node on that line, and inspects the value argument:

```python
record_result("exp1.K2.test_acc", test_acc)   #  computed  → PASS
record_result("exp1.K2.test_acc", 0.816)      #  literal   → FAIL
```

This is "the variables are real" in a form a machine can check.
It is Gate 1's strongest claim, and no prior work we reviewed makes it.

### 3.4 Feedback report

Short, specific, and actionable.
Gate 1 renders it for the agent and also writes it as JSON.

```
GATE 1 — EXECUTION VALIDITY: FAIL   (attempt 2 of 3)

FAILED CHECKS

  [exec.no_uncaught_exception]
    NameError: name 'hidden_dim' is not defined
      run_experiments.py:337  in GCN.forward
      337 |         h = self.lin1(x).view(-1, hidden_dim)

  [results.contract_present]
    No results.json was written. The run crashed before any
    record_result() call completed.

REQUIRED FIXES
  1. Bind `hidden_dim` before use, or pass it into GCN.__init__.
     It is read at line 337 but never assigned in any enclosing scope.
  2. Record every metric named in the plan with record_result(key, value).

STDOUT: 14,208 bytes captured in full (artifacts/attempt_02/stdout.txt)
```

The report holds no score, no praise and no model opinion.
It holds only facts the runtime established.
The em dash in the first line is part of `GATE_NAME`, a published identifier that the signed report package also contains.

#### One model seam, shared by all three gates

Gates 2 and 3 each need a model for the part of their work that has no checkable fact: Gate 2's semantic coherence and Gate 3's claim entailment.
The same rule binds both.
The model sits outside the verdict, and its output is a rate, never a guarantee.

So there is one seam, `consult_model` on each gate's config, carrying one injected `(prompt, system) -> str` callable.
A host wires it once.

| | |
|---|---|
| what runs it | the host's own client, or Ollama with `qwen3-8b-local` in the model registry |
| why local is affordable | the gates make about a third of a phase's calls but use about a sixth of its tokens, and their prompts are short structured requests |
| why it is safe | no verdict consults a model, so a wrong answer degrades a report and never a decision |
| default | the hosted API; local mode is opt-in through `--gate-backend` or `GATE_BACKEND` |

Running the gates on a small local model means the validity layer costs no API budget, so it can check every attempt instead of a sample.

#### The report is generated, and that is why Gate 1 has an LLM layer

The example above is what the feedback report must be, and no template produces it.
"Bind `hidden_dim` before use, or pass it into `GCN.__init__`" depends on where the name is read, which scope it is in, and what the enclosing class takes.
Those facts are in the `GateReport`, and they combine differently on every rejection.
A canned sentence per check id is not a feedback report.

So the LLM layer is part of the Gate 1 pipeline, not an add-on.
It is the loop's way back to the ML engineer.
It does two jobs, and where it sits makes both safe:

```
static → execution → runtime → log → results checks
                                      ↓
                      LLM log scan  (WARN-severity only)
                                      ↓
                                 decide()        ← verdict fixed here, zero LLM
                                      ↓
          PASS → evidence bundle        FAIL → LLM generates the feedback report
```

Neither job can move a verdict.
Log findings are WARN, and `decide()` fails only on a blocking check.
Report generation runs after the verdict exists.
R1.9, "the verdict is computed with no LLM in the path", holds exactly as stated.
§2.1's first invariant holds too: writing the fix for a human-readable report is a place with no checkable fact.
Three further rules constrain both jobs.

- **Grounding.** Generated text may name only check ids, names, line numbers and source lines present in the `GateReport`, and a mechanical check verifies this before the report goes back. A fix that invents a variable name costs the engineer a rewrite, which is the failure Gate 1 exists to prevent, reintroduced at its exit.
- **Declared degradation.** When the model is unavailable, the layer falls back to the deterministic template and says so. A degraded report must never pass for a full one.
- **Verbatim findings.** FAILED CHECKS and WARNINGS are runtime facts and keep their deterministic rendering. The model writes the fixes, not the findings.

Whether the generated report converges the loop faster is a measurement, not an assumption.
`python -m rig.tuning --backend <model>` runs both arms (D61), and [`GATE1_COMPLETION.md`](GATE1_COMPLETION.md) records the history.

### 3.5 What Gate 1 changes downstream

- `exp_results` stops being a 1,000-character prefix. It becomes an evidence bundle: the parsed `results.json`, the full stdout path, and the provenance record.
- `get_score()` survives as a tie-break among attempts that already passed Gate 1. It can no longer admit anything.
- `divergence.jsonl` collects `{attempt, gate1_verdict, failed_checks, reward_score}` for every attempt, and the paper's central evidence table comes from it.

---

## 4. Gate 2, source-result coherence

> **Question:** are these measured results consistent with what the cited literature reports for this method, dataset and setting? Could a reader get these numbers from these sources?

**Position:** after Gate 1, before results are finalized for interpretation and writing.
**Loops back to:** the ML engineer agent, told to revise the plan and the code toward the sources.
**Retry budget:** 2, chosen at setup.
**On exhaustion:** proceed, carrying every unresolved discrepancy forward as a declared limitation that Gate 3 makes the report state.

Gate 2 is where the layer stops being purely deterministic, and the design says so.

> **As built (09-16).** The semantic tier below was removed (D4), and Gate 2 needs no model (D19).
> Gate 2 has three tiers.
> Tier A checks boundaries: ranges, declared relations, and plausibility.
> Tier B checks the run against the plan fields declared at wiring time (D13, D17), and since 09-21 a model may extract those fields, capped at WARN (D55).
> `reference_interval` compares against declared sources bound to `lit_review` (D23).
> Tier C is `review_loop` in `gates/pipeline.py`, and every revision runs under Gate 1 first (F12).
> `progress.md` holds the decision log.
> The text below is the original design.

### 4.1 Inputs

- Gate 1's verified registry: values only, with provenance.
- The literature-review corpus the scaffold already retrieved (`phd.lit_review`, arXiv ids and full text).
- The plan's declared claims.

### 4.2 Checks

The deterministic tier runs first, with no model:

| ID | Check | Severity |
|---|---|---|
| `coherence.range_valid` | Every metric lies in its declared type's admissible range: accuracy in [0,1], time > 0, loss ≥ 0 | FAIL |
| `coherence.internal_consistency` | Declared arithmetic relations hold; a reported speedup equals the ratio of the two recorded times, to tolerance | FAIL |
| `coherence.reference_interval` | Where a source reports a comparable number, the measured value falls inside a stated tolerance band | WARN, or FAIL in strict mode |

The band comes from CORE-Bench's method: the tolerance derives from a reported prediction interval, not from a hand-picked number.
Where a source gives only a point estimate, the band is a declared relative tolerance, recorded as one.

The semantic tier was model-assisted and reported as a rate:

| ID | Check | Severity |
|---|---|---|
| `coherence.method_match` | The implemented method matches the cited source: same normalization, same splits, same objective | WARN |
| `coherence.claim_supported` | The measured values entail each plan claim the results are said to establish | WARN |

### 4.3 Feedback report

It names the metric, the source, and the size of the discrepancy:

```
GATE 2 — SOURCE ↔ RESULT COHERENCE: FAIL   (attempt 1 of 2)

  [coherence.internal_consistency]
    exp2.speedup recorded as 13.61x
    but exp2.sgc.wallclock_s = 0.0180 and exp2.gcn.wallclock_s = 0.2450
    → 0.2450 / 0.0180 = 13.61x   ✓ consistent

  [coherence.reference_interval]
    exp3.noloop.K8.test_acc = 0.3920
    Source: Wu et al. 2019 (arXiv 1902.07153) reports no self-loop ablation at K=8.
    → No source in the retrieved corpus supports a number for this setting.
       Either cite a source that does, or mark this result as novel rather
       than as a replication.
```

### 4.4 Honesty boundary

Gate 2's deterministic tier is provable.
Its semantic tier was not, and the paper must not claim it is.
The planned wording was: "Gate 2's deterministic checks eliminate range and internal-consistency violations by construction. Its semantic tier flags method mismatches at rate R (95% CI …), measured against human annotation on N runs."

---

## 5. Gate 3, report validity

> **Question:** does every number and every citation in the manuscript trace to something that exists?

**Position:** inside the report-generation loop, in place of the LLM reviewer as gatekeeper.
**Loops back to:** the report-writing agent.
**Retry budget:** 3, chosen at setup.
**On exhaustion:** raise `GateFailure`. The gate does not emit a manuscript it cannot verify.

### 5.1 Checks

Numeric binding is deterministic and eliminates fabricated results by construction:

| ID | Check | Severity |
|---|---|---|
| `report.no_numeric_literals_in_results` | No bare numeral appears in a results context; the writer emits only `\result{exp1.K2.test_acc}` tokens | FAIL |
| `report.all_tokens_resolve` | Every `\result{...}` key exists in the Gate 1 registry; an unknown key fails the build | FAIL |
| `report.rendered_values_match_registry` | After rendering, every substituted value is byte-identical to the registry value | FAIL |
| `report.claim_chains` | Each rendered claim's chain, task to command to log to value to claim, written to `claims.json` and reported as a rate. Evidence only, since a run with no task reference is still honest | INFO |

The renderer writes the numbers, not the model.
A number nobody measured has no token, and a token with no value does not render.

Citation binding is deterministic, and it detects rather than constructs (D43):

| ID | Check | Severity |
|---|---|---|
| `source.cited_papers_in_registry` | Every cited arXiv id is in the retrieval registry, a paper the scaffold fetched | FAIL |
| `source.identifiers_resolve` | Every cited id resolves to a paper that exists, through an injected resolver. Absent, and reported absent, when no resolver is supplied or its source is unreachable | FAIL |

Earlier drafts named `report.bibliography_generated` and `report.citation_metadata_matches` and claimed citations were eliminated by construction.
Neither is built, and neither can be on the reference host.
Agent Laboratory writes citations inline as `(arXiv 2308.11483v1)` and emits no bibliography, so a renderer has nothing to generate from the registry and no metadata block to compare against (D26).
Making the construction claim true would mean changing how the host writes citations, which gives up the portability this project argues for.
The two checks above detect a fabricated citation and report a rate instead.
`source.identifiers_resolve` needs no run, so `rig/posthoc_audit.py` also runs it over papers other systems released (figure 7 in `paper/PLAN.md`).

Claim entailment is model-assisted and reported as a rate:

| ID | Check | Severity |
|---|---|---|
| `report.model_unbound_claims` | A model reads the findings prose the number scanner passed and flags what reads like an unbound quantitative claim. Quote-grounded, and it cannot move the verdict | WARN |
| `report.figures_referenced_exist` | Every `\includegraphics` target exists on disk and the gated run produced it | FAIL |

MiniCheck was deferred past this paper (D6).
The WARN above is Gate 1's model layer, rebuilt for manuscripts (D31).

### 5.2 Honesty boundary

Revised against what was built (D43).
Two of the four MLR-Bench classes are eliminated by construction, and the other two are detected and measured.
The difference matters more than the count.

| MLR-Bench class | Gate 3 status |
|---|---|
| Fabricated numeric results | Eliminated by construction, as a property of the pipeline and not of a scanner. See below |
| Silent failure scored as success | Eliminated by construction by Gate 1: a crashed run cannot reach writing |
| Fake or misattributed citations | Detected and measured. The gate rejects a citation nothing retrieved, or one naming no paper that exists, but it cannot stop the writer typing one |
| Unsupported claims in prose | Reduced and measured. "This demonstrates over-smoothing" cannot be made impossible |

For the numeric class, "eliminated by construction" is a property of the pipeline.
The writer emits `\result{key}` tokens and the renderer substitutes registry values, so a number nobody measured has no token and a token with no value does not render.
`report.no_numeric_literals_in_results` checks that the writer used the pipeline, and as a scanner over prose it has a false-negative rate.
The paper must state the construction claim in these terms, with that rate beside it, or it commits the overclaim this gate exists to catch.

**G3-M4, measured** by `rig/gate3_scanner_miss.py`, with labels reviewed and accepted on 2026-09-16 (D37):

| | Gated | Ungated | Both |
|---|---|---|---|
| Claims hand-labelled in findings sections | 39 | 10 | 49 |
| Detected by the scanner | 28 | 6 | 34 |
| Missed | 11 | 4 | 15 |
| Reported but not a claim | 1 | 5 | 6 |

Two manuscripts give counts and a raw fraction, and no confidence interval (D39).
The scanner misses 15 of 49.
The causes matter more than the rate:

| Misses | Cause |
|---|---|
| 12 | The findings line carries a `\ref`, which `SKIP_LINE` matches, so every number on it is dropped and the line reports nothing |
| 2 | `NUMBER` needs a decimal point or four digits, so a two- or three-digit integer result is invisible |
| 1 | `context_of` uses `line.find`, so a value stated twice on one line is counted once |

One cause explains four fifths of what the scanner cannot see, and it is not the one the literature readout predicted.
That probe found the class through `\cite`.
In a real manuscript a findings sentence points at the table or figure it discusses, so `\ref` is the common trigger.
The third row undercounts literals without letting a line through, because the first occurrence still reports.

The paper must keep two scoping notes.
Six of the 40 numbers the scanner reports are not claims: a value quoted from the cited literature, a step index, two thresholds fixed before the run, and a step count the paper specifies.
And the unreadable `\begin{abstract}` (D40) costs nothing on this corpus, because the ungated run recorded no metrics and its abstract states none.
The gap is real, it hid nothing here, and saying otherwise would be the overclaim again.

The scanner stays as it is (D38).
The published Gate 1 traceability number came from it reading `.tex`, so a fix would restate a measured result.

Saying this precisely is worth more than claiming construction on all four classes.

---

## 6. Package layout

G.A.T.E.S. lives in its own repository, apart from any scaffold it is applied to.
That separation is the claim.
A validity layer that can only be described in terms of one research agent is not portable, and portability is what the paper argues.

```
gates/                          standalone repo, pip-installable, zero runtime deps
  pyproject.toml                packages gates* only; ruff rules; the live test marker
  README.md                     what is measured, with figure 8 and the test table
  CLAUDE.md                     the working rules
  progress.md                   state, decision log, session log (STATE is test-checked)
  docs/PLAN.md                  this document
  docs/GATE1_*.md               Gate 1's summary, requirements trace, completion record
  docs/research/                Gate 2 and Gate 3 evidence write-ups
  gates/
    __init__.py                 public API
    errors.py                   GateFailure and friends
    schema.py                   results.json contract, MetricRecord, GateReport, CheckResult
    static_checks.py            symtable scope analysis, literal detection, banned calls
    log_checks.py               error signals in the captured streams
    harness.py                  runs in the child process; injects record_result, dumps results.json
    runner.py                   subprocess execution, process-group kill, full capture, run and trace ids
    gate1.py, gate2.py, gate3.py  the checks and the verdict, one file per gate
    registry.py                 the citable value registry and its provenance chains (CHAIN_LINKS)
    report.py                   feedback report rendering, text and JSON
    llm.py, llm_*.py            the model seam and the layers that sit outside the verdict
    ledger.py                   divergence.jsonl
    setup.py                    retry budgets chosen at wiring time, with the cost warning
    pipeline.py                 the loops every host shares, GATES_LEVEL (D58), level 0's upstream_view
    adapters/
      agentlab.py               reference adapter: Agent Laboratory's defaults and paths only
      arxiv.py                  arXiv resolver for Gate 3; opens a socket, so not in gates/ proper (D41)
  rig/                          model-free loops and evaluations, plus live tools (D61)
  paper/                        the evaluation plan, results.csv, mechanism.csv, figures, collect.py, the draft
  reports/                      the signed 08-15 submission package, frozen
  skills/                       install-gates router and one skill per gate (D59)
  .claude-plugin/               packages skills/ for /plugin install
  tests/                        the suite; conftest.py refuses every socket (D61)
```

The host depends on the package, and the package never depends on the host.
In Agent-Researcher the dependency is a few import lines and the call sites they feed:

```
mlesolver.py     gated_execute() replaces execute_code(); Gate 1 gates get_score()
ai_lab_repo.py   make_context() per phase; the evidence bundle replaces exp_results
tests/test_gates_integration.py   the wiring, tested where the wiring lives
```

Porting to a new scaffold means writing one adapter and changing nothing else in `gates/`.

### 6.1 Adapter contract

An adapter does exactly four things.

1. Build the gate configs and a `GateContext` for the phase.
2. Call `gated_execute` wherever the scaffold used to call its own `exec` helper.
3. Charge one rewrite per agent turn, not per execution. A scaffold with an inner repair loop otherwise spends the agent's whole budget before the agent gets a turn. We hit this while wiring the reference adapter, and the paper should say it: the budget is counted in agent turns.
4. Hand `render_feedback(report)` back to the agent on rejection, and record the attempt in the ledger either way.

---

## 7. Build order

State as of 2026-09-26.
`progress.md` holds the commit for each step.

| Step | Deliverable | State |
|---|---|---|
| 1 | `gates/` package, Gate 1 complete | done |
| 2 | `adapters/agentlab.py` and the MLE-solver wiring; retire `execute_code`'s truncation | done |
| 3 | `divergence.jsonl` and the reward-against-gate evidence table | done |
| 3a | `rig/`, the gate driven turn by turn with no model | done |
| 4 | Fix `run_experiments.py --yaml-location` and re-run the archive for a real n | CLI fixed. The re-run is replaced by the benchmark evaluation (D60) |
| 5 | The channel-fidelity experiment: fabrication rate against `MAX_LEN` | Detector arm measured offline by `rig/reward.py`. Writer arm not scheduled |
| 6 | Gate 2 | Built: tiers A and B, and the loop through Gate 1. Evaluated offline: tier A 27/27 and 0/18, tier B 29/29 |
| 7 | Gate 3 | Built: `report.*` with claim chains, `source.*`, `style.*`, the loop and the model layer. Scanner miss measured, 34 of 49 |
| 8 | `GATES_LEVEL`, the shared loops in `gates/pipeline.py`, per-gate install skills | done (D58, D59) |
| 9 | The benchmark evaluation: CORE-Bench, MLR-Bench and BadScientist at four levels (D60) | Planned in `paper/PLAN.md`. Its tooling is built (`paper/collect.py`, `rig/posthoc_audit.py`, `rig/stats.py`); the host's four-level runner is next, then the pilot |

Step 8 of earlier versions named a CORE-Bench subset and PaperBench Code-Dev.
D60 replaced both with the three benchmarks above.

Two corrections to what this section once said, both checked:

- **`--yaml-location` is fixed.** `run_experiments.py` no longer emits it, and `build_command` produces no argument that `ai_lab_repo.py`'s parser rejects. A quieter defect remained, five YAML keys dropped silently, and host commit `6eb31bf` fixed it: every configured key now passes through, or the config is refused.
- **The channel sweep's detector arm needs no run.** Whether the host's marker survives a given `MAX_LEN` is a property of the capture, and `rig/reward.py` measures it offline. The fabrication-rate arm would still need real runs and a real writing agent.

### 7.1 What Gate 1 would have done to the archived run

The honest version is narrower than the tempting one.

- `static.no_unbound_names` rejects the crashed attempts in the log before execution, since `hidden_dim` is read inside `forward` and bound nowhere. We checked this against a reconstruction of that exact shape.
- The saved `research_dir/src/run_experiments.py` is statically clean, because it is the last successful `REPLACE`, not the crashed one. Gate 1 rejects it on `results.contract_present` instead: it makes no `record_result` call, so no number in the paper is citable.
- So the claim is "Gate 1 rejects this run", not "Gate 1's static tier catches this file". Different checks fire on different artifacts of the same run, and the paper should say which.

---

## 8. Metrics and evaluation protocol

Brought in from `GATE2_implementation_spec.md` §4 and §5 so this repository can state its own experiment.
That spec lives outside the repository, and every "spec §5" in `progress.md` used to point at nothing a checkout contained (D57).

### 8.1 The protocol

D58-D60 in `progress.md` replaced it on 09-21.
The evaluation is three benchmarks, CORE-Bench, MLR-Bench and BadScientist, each run at the four cumulative `GATES_LEVEL`s and each reported as one integrity metric and one task score.
`paper/PLAN.md` is the single source for the protocol, the figures, the significance test (§9) and the steps to run it.
What follows here still applies to it.

### 8.2 The mechanism metrics

These measure how the gates behave, not what they change on a benchmark.
They go in the paper's "Mechanism evidence" appendix beside the `rig/` results, and figure 8 draws the ones already measured.
M2 is the exception: it became the MLR-Bench integrity panel in `paper/PLAN.md`.

| ID | What it reports | Rule |
|---|---|---|
| M1 | construction-level exclusions | Report what cannot happen, never a rate. A perplexity below 1.0, an accuracy outside [0,1] and an arithmetic contradiction among declared relations cannot survive tier A. No interval, because a rate here understates a guarantee |
| M2 | per-type detection rate | `tasks_where_gate_flagged / tasks_where_type_present`, against MLR-Bench's published baselines. A Wilson interval on our two columns only. Their published numbers get none, because we did not measure them |
| M3 | conformance outcomes | Three counts over declared fields, never merged: conforming, divergent, unverifiable. "The run did something else" and "nobody can tell" are different failures |
| M4 | false positives on legitimate runs | `clean_runs_flagged / clean_runs`, with a Wilson interval. Without it the gate looks free, and it is not: a range that rejects a genuine outlier costs a real result |
| M5 | loop behaviour | Resolution rate, mean attempts to resolution, and runs that spent the budget and proceeded with a caveat. The last is intended, and the text must say so or a reviewer reads it as a failure |
| M6 | cost | Wallclock and dollars per run at each level, from each run's manifest, totalled by `paper/collect.py` in `paper/costs.csv`. Gate 2 makes no model call; MLR-Judge needs one per evaluation |

### 8.3 The reporting rule

Every rate carries a Wilson interval and its denominator.
A number without both is a construction-level claim and must be worded as one, never as a measured frequency.

### 8.4 Where the ground truth comes from

Two inputs are neither measured nor deterministic.
Both are now model-authored, so a campaign can run unattended (D54, D55).

- **M2's labels**, which defect types a run contains. A model judge labels them.
- **Gate 2 tier B's `plan_fields`**, what the plan declared. A model extracts them from the host's free-text plan, in the host adapter, before the run. Built 09-21 (F2): `extract_plan_fields` in `gates/adapters/agentlab.py`, run by the host's `--judge-backend` model from level 2, keeps only fields whose quote appears in the plan.

`gates/` still reads no plan and holds no default, and a model still cannot decide a verdict.
What changed is that the input to a deterministic check can be model-authored.
That is weaker than a human declaration, and every report of it must say so.

Two rules keep it honest.

- **A model-authored field cannot fail a run.** A divergence on one is capped at WARN, and the report says the field was model-authored. A human-declared field still FAILs, and a set holding both FAILs on the human field. "A model call can never block" stays literally true.
- **The judge is not the model under test.** Labels and plan fields come from a separate predefined agent. One model writing the plan, extracting the declaration and judging the defect would mostly measure its own self-consistency. D62 names the judges: Gemini Pro and Claude Sonnet, averaged, as MLR-Judge does.
