# G.A.T.E.S. — A Portable Validity Layer for Autonomous Research Agents

**Status:** design locked, Gate 1 in implementation
**Scope:** three deterministic-first gates inserted into an existing autonomous-research scaffold
**Host scaffold (testbed):** Agent-Researcher (this repo), a fork of Agent Laboratory
**Portability target:** AI-Scientist-v2, Agent Laboratory upstream, ScientistOne-class systems

---

## 0. The claim

> A measurable share of hallucinated experimental results in autonomous research agents is
> caused not by model dishonesty but by **lossy or absent evidence channels inside the
> scaffold**. Three deterministic gates placed at the channel boundaries eliminate three of
> the four MLR-Bench hallucination classes by construction, and reduce the fourth to a
> measured rate.

The gates are the contribution. The auto-researcher is the testbed.

### Why this is not already solved

| System | What it does | What it leaves open |
|---|---|---|
| AutoResearchClaw (arXiv 2605.20025) | Verified value registry built during execution | Passes zero-valued results — real measurements, degenerate science |
| SAGE (arXiv 2606.31478) | Grounded reporting; redacts hallucinated numbers | Gates the *output*; writer is still evidence-starved |
| ScientistOne (arXiv 2605.26340) | Chain-of-evidence per claim | No protocol audits whether the claim is *supported* |

All three treat fabrication as a model tendency and catch it at the output. **G.A.T.E.S. treats
it as an information-flow defect and closes the channel at three points.** Gate 1 is upstream of
every published approach: it fires *before* the reward model, *before* interpretation, *before*
writing.

---

## 1. Baseline: what the host scaffold actually does today

Every number below is verified in this repository, not inferred.

### 1.1 The evidence bottleneck

`tools.py:383`

```python
def execute_code(code_str, timeout=60, MAX_LEN=1000):
    ...
    return output_capture.getvalue()[:MAX_LEN]
```

That 1000-character prefix is the *entire* experimental record. It flows:

```
execute_code() ──► args[1] ──► best_codes[i][2] ──► exp_results
                                                       │
   ai_lab_repo.py:347 ──────────────────────────────────┘
                                                       ▼
   papersolver.py:553  "After running this code, the following results were observed: {exp_results}"
```

The writing agent is asked for a results table and given section headers with no values.

### 1.2 The bottleneck also blinds the error detector

This is the finding that upgrades the diagnosis. In `execute_code`, the error marker is written
to the capture buffer **after** the program's own output:

```python
except Exception as e:
    output_capture.write(f"[CODE EXECUTION ERROR]: {str(e)}\n")
    traceback.print_exc(file=output_capture)
```

Once the experiment has printed more than 1000 characters — which every real experiment does —
the marker falls off the end of the slice. And the only crash test in the solver is a substring
search for that marker (`mlesolver.py:91`):

```python
if "[CODE EXECUTION ERROR]" in code_ret: return False, (None, code_ret,)
```

So the crash is invisible. `results/gemini_3_5_flash_run_1/`, log lines 700–745:

```
NameError: name 'hidden_dim' is not defined
$$$$ CODE REPLACE (success)
$$$ Score: 0.98
Running experiments completed, reward function score: 1.0
```

**One line of code causes both the silent failure and the fabrication.** They are not separate
bugs.

### 1.3 Execution has no isolation

`tools.py:402` — `exec_globals = globals()`

Every experiment executes into `tools.py`'s own module namespace, which is never cleared between
attempts. Consequences:

- A name bound by attempt *N* is still bound in attempt *N+1*. An experiment can pass because a
  previous experiment defined a variable. This is precisely the "are the variables real"
  question.
- Generated code can rebind `tools.py` internals (`sys`, `os`, `traceback`, `execute_code`).
- `stderr` is never redirected, so anything the experiment prints there is lost entirely.
- Execution runs in a `ThreadPoolExecutor`; `future.result(timeout=...)` returns but **cannot
  kill the thread**. A runaway experiment keeps burning CPU for the rest of the run, and
  `sys.stdout` is swapped process-globally while it does.

### 1.4 The only quality gate is a model grading a model

`mlesolver.py:151` — `get_score()` asks an LLM at `temp=0.6` for a float in `[0,1]`. Validity is
tested by `float()` parsing, not by whether the run worked:

```python
score, cmd_str, is_valid = get_score(...)
if is_valid:
    failed = False
    break
```

`is_valid` means *a number was returned*. The crashed run above scored 0.95 → 0.98 → 1.0.

### 1.5 Archive state

Seven runs are archived under `results/`. **One completed.** Five died at
`ai_lab_repo.py: error: unrecognized arguments: --yaml-location`; one on a `gemini-2.5-flash-lite`
404. The audit baseline is therefore n=1, not n=7 — fresh runs are required before any rate is
defensible, and `run_experiments.py` has a live argument-passing bug to fix first.

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
                    ║              get_score()  — tie-break only      ║
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

### 2.1 Design invariants

These hold for all three gates and are what make the layer portable.

1. **Deterministic first.** A gate fails on a mechanically checkable fact wherever one exists. A
   model is consulted only where no such fact exists (Gate 1's log recall and feedback-report
   prose, Gate 2's semantic coherence, Gate 3's prose entailment), and where consulted, its
   output is **reported as a rate with a confidence interval, never claimed as a guarantee**.
   Note that "deterministic first" constrains where a model sits, not whether one is present:
   every gate's *verdict* is deterministic, and Gate 1's LLM layer is required precisely because
   it is positioned outside that verdict (§3.4).
2. **The gate is the authority.** No model opinion can admit an artifact the gate rejected.
3. **Every gate emits two artifacts:** a machine-readable verdict (`gate<N>_report.json`) and a
   small natural-language feedback report for the agent it loops back to.
4. **Bounded loops.** Every gate has a retry budget. Exhaustion has a defined terminal state; no
   gate can spin forever.
5. **Zero scaffold imports.** `gates/` imports nothing from the host scaffold. All host knowledge
   lives in `adapters/`.
6. **Evidence is append-only.** Gates never edit agent output. They accept, reject, or annotate.

### 2.2 Severity model

| Severity | Effect |
|---|---|
| `FAIL` | Gate verdict is FAIL. Artifact rejected, feedback loop fires. |
| `WARN` | Recorded in the report and surfaced to the agent; does not block. |
| `INFO` | Provenance only. |

Verdict = `FAIL` if any check is `FAIL`, else `PASS`.

---

## 3. GATE 1 — Execution Validity

> **Question:** Did this code actually run to completion, and were the numbers it reports
> produced by *this* run rather than inherited, hardcoded, or invented?

**Position:** between code execution and the reward model, inside the MLE solver loop.
**Loops back to:** the ML engineer agent.
**Retry budget:** 3 rewrites.
**On exhaustion:** fall back to the most recent attempt that passed Gate 1. If no attempt ever
passed, raise `GateFailure` and exit the run non-zero. No paper is produced from an unexecuted
experiment.

### 3.1 Requirements

| # | Requirement | Rationale |
|---|---|---|
| R1.1 | Experiment code executes in a **fresh OS process** with an empty `__main__` namespace | Kills the `exec_globals = globals()` leak; makes "real variables" structurally checkable |
| R1.2 | `stdout` and `stderr` captured **separately and in full**, to files, never truncated | Retires `MAX_LEN=1000`; restores the evidence channel |
| R1.3 | Failure detection reads the **process exit code**, not a substring of stdout | Removes the coupling that made truncation hide crashes |
| R1.4 | Timeout kills the **entire process group** | A runaway experiment must not survive its own timeout |
| R1.5 | Experiments declare results through a **typed contract** (`results.json`), not by printing | Downstream consumers stop parsing prose |
| R1.6 | Every declared metric records **call-site provenance** (file, line, source text) | Enables the anti-literal check and Gate 3's chain of evidence |
| R1.7 | A metric whose value is a **source literal** fails the gate | A typed number is not a measurement |
| R1.8 | Unbound names are detected **statically, before execution** | A 45-minute run should not be spent to discover a `NameError` |
| R1.9 | Gate verdict is computed with **no LLM in the path** | The runtime already knows whether the run succeeded |
| R1.10 | Every attempt is logged to `divergence.jsonl` with both the gate verdict and the reward score | The disagreement between them is a headline result of the paper |
| R1.11 | Both captured streams are scanned for **errors the run continued past**, reported and never blocking | The spoken spec: "log files without errors that don't seem really code-breaking". A clean exit code cannot see a caught exception, a divide-by-zero NaN, or a CUDA fallback |
| R1.12 | Every recorded value is **bound to one execution** by a trace id, and the executed source is hashed by the process that ran it | Deck slide 11: "typed and hashed to the run that produced it". Without the second half, the binding is an assumption |
| R1.13 | Each value carries its **provenance chain link by link**, with unresolved links reported rather than omitted | ARGUS Sub-Topic C: a value that merely matches a log is a worse signal than a numeric discrepancy, so the two must be distinguishable |
| R1.14 | **Every** `record_result` call is retained, not only the last | ARGUS Sub-Topic B's observed failure mode: a best-epoch number reported as a final-epoch one is invisible if earlier calls are discarded |
| R1.15 | A run that declares **no seed** is reported as unreproducible | Execution-grounded verification — re-running the code to check the claim — is impossible without it |

Full traceability from each source document's requirements to the check and test that satisfy
them is maintained separately in [`GATE1_REQUIREMENTS.md`](GATE1_REQUIREMENTS.md), including the
one requirement that is only partially met (P5, undisclosed replicate drops).

### 3.2 Checks

Static — run before execution, so a rejection costs no compute:

| ID | Check | Severity |
|---|---|---|
| `static.syntax_valid` | Source compiles | FAIL |
| `static.no_unbound_names` | Scope analysis finds no name loaded but never bound and not a builtin/import | FAIL |
| `static.no_banned_calls` | No `exit()`, `sys.exit()`, `os._exit()` — they forge a clean exit code | FAIL |

Runtime:

| ID | Check | Severity |
|---|---|---|
| `exec.exit_code_zero` | Subprocess exited 0 | FAIL |
| `exec.no_uncaught_exception` | Harness recorded no escaping exception | FAIL |
| `exec.completed_within_budget` | Not killed by timeout | FAIL |
| `env.clean_namespace` | Harness asserts the initial global namespace contained only harness-provided names | FAIL |
| `env.code_identity` | The source hashed by the child process matches the source the parent wrote | FAIL |
| `exec.no_swallowed_traceback` | No `Traceback` in **either** stream from a run that still exited 0 | WARN |
| `logs.no_error_signals` | No numerical-integrity warning, device failure, non-convergence, printed exception or `ERROR`-level log | WARN |
| `env.seed_recorded` | The experiment declared a seed | WARN |

Results contract:

| ID | Check | Severity |
|---|---|---|
| `results.contract_present` | `results.json` exists and parses | FAIL |
| `results.expected_keys_present` | Every key the plan declared is present (strict mode) | FAIL |
| `results.values_computed` | No metric value is an `ast.Constant` at its record call site | FAIL |
| `results.values_finite` | No `NaN`, no `±inf` | FAIL |
| `results.single_observation` | No key was recorded repeatedly with a changing value | WARN |
| `results.non_degenerate` | Flags exact `0.0`, exact `1.0`, and exact chance level when class count is known | WARN |

`results.non_degenerate` is the check that answers AutoResearchClaw's stated limitation — a
registry alone passes real zeros. We surface them rather than silently accepting them.

`logs.no_error_signals` is the WARN tier the spoken spec asks for: errors present in the logs that
did not break the run. Its pattern set is tuned for precision, because a false positive costs the
agent a rewrite for nothing — `ValueError:` is flagged and `Mean Squared Error:` is not, and both
directions are held by test. Its recall is explicitly unbounded, and that limitation is stated
rather than hidden.

`results.single_observation` exists because the harness now retains every `record_result` call.
The gate cannot know which of an epoch loop's values the paper intends, so it reports the span and
requires the report to disambiguate.

Provenance (recorded, not gated):

| ID | Recorded | Severity |
|---|---|---|
| `output.untruncated` | stdout/stderr byte counts; asserts no truncation was applied | INFO |
| `env.provenance` | Python version, platform, seed, device, key library versions, code SHA-256 | INFO |
| `env.parent_proc_guard` | whether the experiment could read the gate's own process: `active`, or why not (`bypassable` under CAP_SYS_PTRACE, `failed`, `unsupported`) | INFO |

### 3.3 The results contract

The experiment declares results through an API the harness injects into its namespace — no
`pip install`, no import path, nothing for the agent to get wrong:

```python
record_result("exp1.K2.test_acc", test_acc, unit="ratio")
record_result("exp2.sgc.wallclock_s", total_sgc_time, unit="seconds")
```

`record_result` captures the caller's frame — filename, line number, and the source text of the
call. Gate 1 then re-parses the source, locates the `Call` node at that line, and inspects the
value argument:

```python
record_result("exp1.K2.test_acc", test_acc)   #  computed  → PASS
record_result("exp1.K2.test_acc", 0.816)      #  literal   → FAIL
```

This is the mechanically checkable form of "the variables are real". It is the strongest claim in
Gate 1 and, as far as the reviewed prior art goes, unclaimed.

### 3.4 Feedback report

Small, specific, actionable. Rendered for the agent; also written as JSON.

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

Note what is *not* in the report: no score, no praise, no model opinion. Only facts the runtime
established.

#### One model seam, shared by all three gates

The layer described below is not Gate 1's private arrangement. Gates 2 and 3 each
need a model for the part of their work that has no mechanically checkable fact —
Gate 2's semantic coherence, Gate 3's claim entailment — and both are subject to
the same rule: the model is consulted only outside the verdict, and its output is
reported as a rate rather than claimed as a guarantee.

So there is one seam, `Gate1Config.consult_model` today and a shared config field
as Gates 2 and 3 land, carrying one injected `(prompt, system) -> str` callable.
A host wires it once. The decided default deployment is **a locally run
`qwen3:8b`** for all three gates, with the hosted API as the fallback:

| | |
|---|---|
| what runs it | Ollama, `qwen3-8b-local` in the model registry |
| why local is affordable here | the gates are ~⅓ of a phase's calls but ~⅙ of its tokens, and their prompts are short structured requests |
| why it is safe | no gate's verdict consults a model; a wrong answer degrades a report, never a decision |
| default | the hosted API — local mode is opt-in via `--gate-backend` or `GATE_BACKEND` |

Running the gates on a small local model is also a claim the paper can make on
its own: the validity layer costs no API budget to operate, which is what lets it
be applied to every attempt rather than sampled.

#### The report is generated, and that is why Gate 1 has an LLM layer

The example above is what the feedback report must be, and no template produces it. "Bind
`hidden_dim` before use, or pass it into `GCN.__init__`" depends on where the name is read, what
scope it is in, and what the enclosing class takes — facts that exist in the `GateReport` but
compose differently on every rejection. A canned sentence per check id is not a feedback report.

So the LLM layer is a **required component of the Gate 1 pipeline**, not an optional enhancement:
it is the loop's return path to the ML engineer. It does two jobs, and its position is what makes
both safe:

```
static → execution → runtime → log → results checks
                                      ↓
                      LLM log scan  (WARN-severity only)
                                      ↓
                                 decide()        ← verdict fixed here, zero LLM
                                      ↓
          PASS → evidence bundle        FAIL → LLM generates the feedback report
```

Neither job can move a verdict. Log findings are WARN, and `decide()` fails only on a *blocking*
check; report generation runs after the verdict already exists. R1.9 — "gate verdict is computed
with no LLM in the path" — holds exactly as stated, and §2.1's first invariant is satisfied
rather than bent: a model is consulted only where no mechanically checkable fact exists, and
writing the fix for a human-readable report is precisely that place. Both jobs are constrained
further:

- **Grounding.** Generated text may reference only check ids, names, line numbers and source
  lines present in the `GateReport`, verified mechanically before the report is returned. A fix
  that invents a variable name costs the engineer a rewrite — the failure Gate 1 exists to
  prevent, reintroduced at its exit.
- **Degradation, declared.** When the model is unavailable the layer falls back to the
  deterministic template and says that it did. A degraded report must never be mistaken for a
  full one.
- **The findings stay verbatim.** FAILED CHECKS and WARNINGS are runtime facts and keep their
  deterministic rendering. The model writes the fixes, not the findings.

Whether the generated report actually converges the loop faster is measured, not assumed — see
`rig/gate1_loop.py` and `GATE1_COMPLETION.md`.

### 3.5 What Gate 1 changes downstream

- `exp_results` stops being a 1000-character prefix. It becomes an **evidence bundle**: the
  parsed `results.json`, the full stdout path, and the provenance record.
- `get_score()` survives as a **tie-break among attempts that already passed Gate 1**. It can no
  longer admit anything.
- `divergence.jsonl` accumulates `{attempt, gate1_verdict, failed_checks, reward_score}` for every
  attempt. The paper's central evidence table is generated from this file.

---

## 4. GATE 2 — Source ↔ Result Coherence

> **Question:** Are these measured results consistent with what the cited literature reports for
> this method, dataset, and setting? Could a reader obtain these numbers from these sources?

**Position:** after Gate 1, before results are finalized for interpretation and writing.
**Loops back to:** the ML engineer agent, with instruction to revise the plan and the code toward
the sources.
**Retry budget:** 2.
**On exhaustion:** proceed, with every unresolved discrepancy carried forward as a mandatory
declared limitation that Gate 3 will require the report to state.

Gate 2 is where the layer stops being purely deterministic, and the design says so openly.

> **As built (09-16).** The semantic tier below was removed (D4), and Gate 2 is model-free (D19).
> Gate 2 has three tiers. Tier A checks boundaries: ranges, declared relations, and plausibility.
> Tier B checks the run against the plan fields declared at wiring time (D13, D17).
> `reference_interval` compares against declared sources bound to `lit_review` (D23).
> Tier C is `review_loop` in the adapter, and every revision runs under Gate 1 first (F12).
> `progress.md` holds the decision log. The text below is the original design.

### 4.1 Inputs

- Gate 1's verified registry (values only, with provenance)
- The literature-review corpus already retrieved by the scaffold (`phd.lit_review`, arXiv IDs +
  full text)
- The plan's declared claims

### 4.2 Checks

Deterministic tier — runs first, no model involved:

| ID | Check | Severity |
|---|---|---|
| `coherence.range_valid` | Every metric lies in its declared type's admissible range (accuracy ∈ [0,1], time > 0, loss ≥ 0) | FAIL |
| `coherence.internal_consistency` | Declared arithmetic relations hold — a reported speedup equals the ratio of the two recorded times, to tolerance | FAIL |
| `coherence.reference_interval` | Where a source reports a comparable number, the measured value falls within a stated tolerance band | WARN → FAIL in strict mode |

The reference-interval band is the principled part and is taken from CORE-Bench's methodology:
tolerance is derived from a reported prediction interval, not chosen by hand. Where a source gives
only a point estimate, the band is a declared relative tolerance and is recorded as such.

Semantic tier — model-assisted, reported as a rate:

| ID | Check | Severity |
|---|---|---|
| `coherence.method_match` | The implemented method matches what the cited source describes (same normalization, same splits, same objective) | WARN |
| `coherence.claim_supported` | Each plan claim the results are said to establish is entailed by the measured values | WARN |

### 4.3 Feedback report

Names the specific metric, the specific source, and the size of the discrepancy:

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

Gate 2's deterministic tier is provable. Its semantic tier is not, and the paper must not claim
it is. Reported as: *"Gate 2's deterministic checks eliminate range and internal-consistency
violations by construction. Its semantic tier flags method mismatches at rate R (95% CI …),
measured against human annotation on N runs."*

---

## 5. GATE 3 — Report Validity

> **Question:** Does every number and every citation in the manuscript trace to something that
> actually exists?

**Position:** inside the report-generation loop, replacing the LLM reviewer's role as gatekeeper.
**Loops back to:** the report-writing agent.
**Retry budget:** 3.
**On exhaustion:** raise `GateFailure`. An unverifiable manuscript is not emitted.

### 5.1 Checks

Numeric binding — deterministic, eliminates fabricated results by construction:

| ID | Check | Severity |
|---|---|---|
| `report.no_numeric_literals_in_results` | No bare numeral appears in a results context. The writer emits `\result{exp1.K2.test_acc}` tokens only | FAIL |
| `report.all_tokens_resolve` | Every `\result{...}` key exists in the Gate 1 registry. An unknown key fails the build | FAIL |
| `report.rendered_values_match_registry` | Post-render, every substituted value is byte-identical to the registry value | FAIL |

The renderer, not the model, writes the numbers. A number that was never measured has no token,
and a token that has no value does not compile.

Citation binding — deterministic, and **detection rather than construction**
(as-built, D43):

| ID | Check | Severity |
|---|---|---|
| `source.cited_papers_in_registry` | Every cited arXiv ID is in the retrieval registry — a paper the scaffold actually fetched | FAIL |
| `source.identifiers_resolve` | Every cited ID resolves to a paper that exists, through an injected resolver. Absent, and reported absent, when no resolver is supplied or its source is unreachable | FAIL |

Earlier drafts of this section named `report.bibliography_generated` and
`report.citation_metadata_matches` and claimed citations were eliminated by
construction. Neither is built, and neither can be on the reference host: Agent
Laboratory writes citations inline as `(arXiv 2308.11483v1)` and emits no
bibliography, so there is nothing for a renderer to generate from the registry
and no metadata block to compare against (D26). Making the construction claim
true would mean changing how the host writes citations, which costs the
portability claim this project argues for. The two checks above detect a
fabricated citation and report a rate instead.

Claim entailment — model-assisted, reported as a rate:

| ID | Check | Severity |
|---|---|---|
| `report.model_unbound_claims` | A model reads the findings prose the number scanner passed and flags what reads like an unbound quantitative claim. Quote-grounded, and it cannot move the verdict | WARN |
| `report.figures_referenced_exist` | Every `\includegraphics` target exists on disk and was produced by the gated run | FAIL |

MiniCheck was deferred past this paper (D6). The WARN above is the model layer
Gate 1 already had, rebuilt for manuscripts (D31).

### 5.2 Honesty boundary

Revised against what was built (D43). Two of the four MLR-Bench classes are
eliminated by construction; the other two are detected and measured, and the
difference matters more than the count:

| MLR-Bench class | Gate 3 status |
|---|---|
| Fabricated numeric results | **Eliminated by construction** — of the *pipeline*, not of a scanner. See below |
| Silent failure scored as success | **Eliminated by construction** (Gate 1) — a crashed run cannot reach writing |
| Fake / misattributed citations | **Detected and measured** — a citation nothing retrieved, or naming no paper that exists, is rejected. The writer is not prevented from typing one |
| Unsupported claims in prose | **Reduced and measured** — "this demonstrates over-smoothing" cannot be made impossible |

**What "eliminated by construction" means for the numeric class.** It is a
property of the pipeline: the writer emits `\result{key}` tokens, the renderer
substitutes registry values, so a number that was never measured has no token
and a token with no value does not render.
`report.no_numeric_literals_in_results` is the check that the pipeline was
*used*, and as a scanner over prose it has a false-negative rate. The paper must
state the construction claim in these terms, with that number beside it, or it is
the same overclaim this gate exists to catch.

**G3-M4, measured** (`rig/gate3_scanner_miss.py`; labels reviewed and accepted
2026-09-16, D37):

| | Gated | Ungated | Both |
|---|---|---|---|
| Claims hand-labelled in findings sections | 39 | 10 | **49** |
| Detected by the scanner | 28 | 6 | **34** |
| Missed | 11 | 4 | **15** |
| Reported but not a claim | 1 | 5 | **6** |

Two manuscripts, so counts and a raw fraction, and no confidence interval (D39).
The miss rate is 15 of 49. The rate is not the finding; the causes are:

| Misses | Cause |
|---|---|
| 12 | The findings line carries a `\ref`, which `SKIP_LINE` matches, so every number on it is dropped and the line reports nothing |
| 2 | `NUMBER` requires a decimal point or four digits, so a two- or three-digit integer result is invisible |
| 1 | `context_of` uses `line.find`, so a value stated twice on one line is deduplicated to one |

Four fifths of everything the scanner cannot see has a single cause, and it is
not the one the literature readout predicted: that probe found the class through
`\cite`, but in a real manuscript a findings sentence points at the table or
figure it discusses, so `\ref` is the common trigger. The third row understates
the literal *count* without letting a line through, since the first occurrence
still reports.

Two scoping notes the paper must keep. Six of the 40 numbers the scanner reports
are not claims at all: a value quoted from the cited literature, a step index, two
thresholds fixed before the run, and a step count the paper specifies. And the
unreadable `\begin{abstract}` (D40) costs nothing on this corpus, because the
ungated run recorded no metrics and its abstract states none — the gap is real
and it hid nothing here, and saying otherwise would be the overclaim again.

The scanner is not changed (D38): the published Gate 1 traceability number came
from it reading `.tex`, so a fix restates a measured result.

Saying this precisely is worth more than claiming construction on all four.

---

## 6. Package layout

G.A.T.E.S. lives in its own repository, separate from any scaffold it is applied to. That
separation is not tidiness — it is the claim. A validity layer that can only be described in
terms of one research agent is not portable, and portability is what the paper argues for.

```
gates/                        (standalone repo, pip-installable, zero runtime deps)
  pyproject.toml
  README.md
  docs/PLAN.md                this document
  docs/GATE1_SUMMARY.md       what Gate 1 does and why
  docs/GATE1_REQUIREMENTS.md  each source requirement → check → test
  gates/
    __init__.py               public API: run_gate1, Gate1Config, GateFailure, registry helpers
    errors.py                 GateFailure and friends
    schema.py                 results.json contract, MetricRecord, GateReport, CheckResult
    static_checks.py          symtable scope analysis, literal detection, banned calls
    log_checks.py             error signals in the captured streams
    harness.py                runs in the child process; injects record_result, dumps results.json
    runner.py                 sandboxed subprocess execution, process-group kill, full capture,
                              run and trace identifiers
    gate1.py                  the checks and the verdict
    registry.py               the citable value registry and its provenance chains
    report.py                 feedback report rendering (text + JSON)
    ledger.py                 divergence.jsonl
    adapters/
      agentlab.py             reference adapter — Agent Laboratory / Agent-Researcher
  tests/
    test_gate1.py             gate behaviour, independent of any scaffold
```

The host scaffold depends on the package; the package never depends on the host. In
Agent-Researcher that dependency is three import lines and the call sites they feed:

```
mlesolver.py     gated_execute() replaces execute_code(); Gate 1 gates get_score()
ai_lab_repo.py   make_context() per phase; the evidence bundle replaces exp_results
tests/test_gates_integration.py   the wiring, tested where the wiring lives
```

Porting to a new scaffold means writing one adapter and changing nothing else in `gates/`.

### 6.1 Adapter contract

An adapter is responsible for exactly four things:

1. Build a `Gate1Config` and a `GateContext` for the phase.
2. Call `gated_execute` wherever the scaffold used to call its own `exec` helper.
3. Charge **one rewrite per agent turn** — not per execution. A scaffold with an inner
   automated-repair loop will otherwise spend the agent's entire budget before the agent gets
   a turn. (This is a real defect we hit while wiring the reference adapter, and it is worth
   calling out in the paper: the budget must be denominated in agent turns.)
4. Hand `render_feedback(report)` back to the agent on rejection, and record the attempt to the
   ledger either way.

---

## 7. Build order

| Step | Deliverable | State |
|---|---|---|
| 1 | `gates/` package — Gate 1 complete | **done** |
| 2 | `adapters/agentlab.py` + MLE-solver wiring; retire `execute_code`'s truncation | **done** |
| 3 | `divergence.jsonl` and the reward-vs-gate evidence table | **done** |
| 3a | `rig/` — the loop rig: the gate driven engineer-turn by engineer-turn, no model | **done** |
| 4 | Fix `run_experiments.py --yaml-location`; re-run the archive for a real n | **CLI fixed**; runs pending |
| 5 | The channel-fidelity experiment: fabrication rate vs. `MAX_LEN` | detector arm instrumented; writer arm pending |
| 6 | Gate 2 | **built**: tiers A and B, loop (tier C) through Gate 1; campaign (E1) pending |
| 7 | Gate 3 | **built**: `report.*`, `source.*`, `style.*`, the loop and the model layer; campaign pending |
| 8 | Evaluate on a CORE-Bench subset and PaperBench Code-Dev | pending |

Steps 4 and 5 produce a result either way. If widening the channel changes the numbers, that is
the finding. If they come back identical to the published SGC paper, that is a stronger one.

The remaining Gate 1 work is measurement, not implementation, and is scheduled in
[`GATE1_COMPLETION.md`](GATE1_COMPLETION.md). Two corrections to what this section
previously said, both verified rather than assumed:

- **`--yaml-location` is already fixed.** `run_experiments.py` no longer emits it, and
  `build_command` produces no argument `ai_lab_repo.py`'s parser rejects. What remains in that
  file is a quieter defect: five YAML keys are dropped silently instead of being passed or
  refused, so a config can declare a lit-review backend the run does not use.
- **The channel sweep's detector arm no longer needs a run.** Whether upstream's marker survives
  a given `MAX_LEN` is a property of the capture, and `rig/reward.py` measures it offline. The
  fabrication-rate arm still needs real runs and a real writing agent.

### 7.1 What Gate 1 would have done to the archived run — precisely

Worth stating carefully, because the honest version is narrower than the tempting one.

- The **crashed attempts** in the log (`hidden_dim` read inside `forward`, bound nowhere) are
  rejected by `static.no_unbound_names` *before execution*. Verified against a reconstruction of
  that exact shape.
- The **saved** `research_dir/src/run_experiments.py` is statically clean — it is the last
  successful `REPLACE`, not the crashed one. Gate 1 rejects it on
  `results.contract_present` instead: it makes zero `record_result` calls, so not one of the
  numbers in the paper is citable.
- So the claim to make is *"Gate 1 rejects this run"*, not *"Gate 1's static tier catches this
  file"*. Different checks fire on different artifacts of the same run, and the paper should say
  which.

---

## 8. Metrics and evaluation protocol

Brought in from `GATE2_implementation_spec.md` §4 and §5 so this repository can state its own
experiment. The spec lives outside the repository, and every reference to "spec §5" in
`progress.md` used to be unresolvable from a checkout (D57).

### 8.1 The six metrics

| ID | What it reports | Rule |
|---|---|---|
| M1 | construction-level exclusions | report what **cannot** happen, never a rate. A perplexity below 1.0, an accuracy outside [0,1] and an arithmetic contradiction among declared relations cannot survive tier A. No interval: a rate here understates a guarantee |
| M2 | per-type detection rate | `tasks_where_gate_flagged / tasks_where_type_present`, against MLR-Bench's published baselines. Wilson interval on our two columns only. Their published numbers get no interval: we did not measure them |
| M3 | conformance outcomes | three counts over declared fields, never collapsed: conforming, divergent, unverifiable. "The run did something else" and "nobody can tell" are different failures |
| M4 | false positives on legitimate runs | `clean_runs_flagged / clean_runs`, with a Wilson interval. Without it the gate looks free, and it is not: a range that rejects a genuine outlier costs a real result |
| M5 | loop behaviour | resolution rate, mean attempts to resolution, and runs that exhausted the budget and proceeded with a caveat. The last is a feature, and the text must say so or a reviewer reads it as a failure |
| M6 | cost | Gate 2 makes zero model calls, so report wallclock overhead per gated run against ungated. MLR-Judge needs a model per evaluation; Gate 2 needs none |

### 8.2 E1, the positive set

MLR-Bench's ten experimentation tasks, run gated and ungated, same seeds, same model, paired.
Report per-task differences, not only aggregates.

Two arms, not four: every gate off against every gate on (D53).
Per-gate arms are a later experiment, and the tier comparison that needs no model already exists
in `rig/gate2_tier_comparison.py`.
`GATES_GATE1=off` is the switch for the whole layer, and the host skips Gates 2 and 3 when it is
set, because neither has an input without Gate 1's registry.

The first run is a pilot: one task, both arms, one seed, all three gates.
It sets the cost and the wallclock per run, which is also M6, and only then is the number of
seeds chosen.
Ten tasks at one seed put a result of 8/10 inside [0.49, 0.94]; thirty runs at 24/30 narrow that
to [0.63, 0.90].

The tasks come from MLR-Bench's own release rather than retyped from the paper.
The runner is the host's `tools_full_gate1_ablation.py`: one parsed config, two arms in isolated
working directories, a manifest per run.
It stays in the host because `rig/` never opens a socket (D41) and never calls a model.

### 8.3 E2, the negative set

BadScientist cannot serve here: its strategies manipulate how a paper presents itself, so it
tests Gate 3, and fabrication happens before a paper exists.
It is held for Gate 3's adversarial evaluation, which is not built (F18).

Gate 2's negatives are synthesized instead, one registry per defect class, each asserting the
specific check id that fires.
A fixture that fails for the wrong reason is a passing test hiding a broken check.
Six of the seven live as cases in `rig/gate2_tier_a_eval.py` and `rig/gate2_tier_b_eval.py`; the
seventh, a random binary baseline at 0.65, was dropped with its reason (D20).

### 8.4 E3, the reporting rule

Every rate carries a Wilson interval and its denominator.
A number without both is a construction-level claim and must be phrased as one, never as a
measured frequency.

### 8.5 Where the ground truth comes from

Two inputs are neither measured nor deterministic, and both are now model-authored so a campaign
can run unattended (D54, D55).

- **M2's labels**: which defect types are present in a run. A model judge labels them.
- **Gate 2 tier B's `plan_fields`**: what the plan declared. A model extracts them from the
  host's free-text plan, in the host adapter, before the run.

`gates/` still reads no plan and still holds no default, and a model still cannot decide a
verdict.
What changes is that the *input* to a deterministic check can be model-authored, which is weaker
than a human declaration and must be labelled as such wherever it is reported.

Two rules keep that honest:

- **A model-authored field cannot fail a run.** A divergence on one is capped at WARN and the
  report says the field was model-authored. A human-declared field still FAILs, and a set holding
  both FAILs on the human field. "A model call can never block" stays literally true.
- **The judge is not the model under test.** Labels and plan fields come from a separate
  predefined agent. One model writing the plan, extracting the declaration and judging the defect
  would mostly measure its own self-consistency.
