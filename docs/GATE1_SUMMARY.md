# Gate 1 — Execution Validity

**One-line summary.** Gate 1 sits between code execution and the reward model and answers a
single question with no model in the loop: *did this code actually run to completion, and were
the numbers it reports produced by this run rather than inherited, hardcoded, or invented?*

Status: **implemented, tested, wired into Agent-Researcher.** 41 gate tests, 10 integration
tests.

---

## 1. The problem it solves

In the audited scaffold (Agent Laboratory / Agent-Researcher), a single line caused both a silent
failure and a fabricated paper:

```python
# tools.py
def execute_code(code_str, timeout=60, MAX_LEN=1000):
    ...
    except Exception as e:
        output_capture.write(f"[CODE EXECUTION ERROR]: {str(e)}\n")   # appended AFTER the prints
    return output_capture.getvalue()[:MAX_LEN]                        # ...then sliced off
```

Two consequences, from the same slice:

1. **The writer was starved.** `exp_results` — the entire experimental record handed to the
   paper-writing agent — was 1000 characters. Real experiments print more than that before
   reaching their metrics, so the writer received section headers with no values and generated
   plausible ones.
2. **The failure detector was blinded.** The crash marker is appended *after* the program's own
   output, so it fell off the end of the same 1000-character slice. The solver's only crash test
   was `if "[CODE EXECUTION ERROR]" in code_ret`, which never fired.

Observed in `results/gemini_3_5_flash_run_1/`:

```
NameError: name 'hidden_dim' is not defined
$$$$ CODE REPLACE (success)
$$$ Score: 0.98
Running experiments completed, reward function score: 1.0
```

That run produced a paper reporting 81.60% test accuracy, a 13.61× speedup, and a 39.20%
ablation collapse — to two decimal places. None of it was measured.

Three further defects compounded it: `exec(code_str, globals())` executed into `tools.py`'s own
never-cleared namespace, so a name bound by one attempt stayed bound for the next; execution ran
in a thread that `future.result(timeout=)` could not kill; and `stderr` was never captured at
all.

---

## 2. What Gate 1 does

### Static tier — before anything runs, so rejection costs no compute

| Check | Fails when |
|---|---|
| `static.syntax_valid` | source does not compile |
| `static.no_unbound_names` | a name is read but bound nowhere |
| `static.no_banned_calls` | `exit()` / `sys.exit()` would forge a clean exit code |

`static.no_unbound_names` uses `symtable`, so closures, comprehensions, `global`/`nonlocal`,
walrus bindings, parameters, imports and class scopes are resolved by the same machinery the
interpreter uses. It is deliberately conservative — module-level use-before-assignment is *not*
reported, because that needs ordering analysis and would produce false positives. Verified
zero false positives across nine valid-scoping patterns; the runtime tier catches what the
static tier declines to claim.

### Runtime tier — fresh process, empty namespace

| Check | Fails when |
|---|---|
| `exec.exit_code_zero` | the process exited non-zero |
| `exec.no_uncaught_exception` | an exception escaped |
| `exec.completed_within_budget` | the process group was killed on timeout |
| `env.clean_namespace` | anything was inherited before execution |
| `exec.no_swallowed_traceback` *(warn)* | the code caught an error, printed it, and continued |

`env.clean_namespace` is what makes "the variables are real" a property of the runtime rather
than a hope: the harness snapshots the initial globals and Gate 1 asserts they contain only
harness-provided names.

### Results contract

Experiments declare results through an API injected into their namespace — no import, nothing
for the agent to get wrong:

```python
record_result("exp1.K2.test_acc", test_acc, unit="ratio")
```

| Check | Fails when |
|---|---|
| `results.contract_present` | nothing was recorded |
| `results.expected_keys_present` | a key the plan declared is missing |
| `results.values_computed` | a recorded value is a source literal |
| `results.values_finite` | a value is NaN or infinite |
| `results.non_degenerate` *(warn)* | exact zero, perfect score, or chance level |

**`results.values_computed` is the strongest claim here.** `record_result` captures its own call
site; Gate 1 re-parses the source, locates the `Call` node at that line, and inspects the value
argument. `record_result("k", acc)` passes. `record_result("k", 0.816)` fails, as does
`float(80.40)/100` and `round(0.8160, 3)` — constant-folding does not launder a typed number.
No reviewed prior art claims this check.

`results.non_degenerate` answers the limitation AutoResearchClaw (arXiv 2605.20025) reports for
value registries: a registry passes zero-valued results because the zeros are genuine
measurements. We surface them and never block, because a legitimate zero exists.

---

## 3. Loop and budget

```
ML engineer writes code
        │
        ▼
   Gate 1 ──PASS──► get_score() ranks ──► best_codes
        │                    │
      FAIL                   └──► divergence.jsonl
        │
        ▼
  feedback report ──► ML engineer rewrites   (max 3 rewrites)
        │
   budget spent:
     any attempt passed?  yes ──► fall back to it, continue
                          no  ──► GateFailure, run exits non-zero, no paper
```

**The budget counts agent rewrites, not executions.** This distinction is load-bearing and was a
real defect caught by the integration tests: the scaffold's inner `code_repair` loop runs several
executions per engineer turn, and denominating the budget in executions meant the automated
repair tool spent all three retries before the ML engineer got a single turn. Any scaffold with
an inner repair loop hits this; it is written into the adapter contract.

**`get_score` is demoted, not deleted.** Gate 1 owns pass/fail; the reward model only orders
what already passed. Rejected attempts are still scored — against a faithful reconstruction of
the 1000-character view the upstream scaffold would have had — and logged. That makes
`divergence.jsonl` a measured counterfactual ("what would the reward model have said, given the
channel the scaffold actually had?") rather than an anecdote, and it is where the paper's
evidence table comes from.

---

## 4. What the writer receives instead of 1000 characters

```
VERIFIED RESULTS — Gate 1 PASS (attempt 3)

Every value below was recorded by record_result() during the run that
produced the code above, and traced back to the line that computed it.
These are the only numbers that may be reported.

  exp1.K2.test_acc     = 0.816  [ratio]  (experiment.py:130)
  exp2.sgc.wallclock_s = 0.018  [seconds] (experiment.py:264)

RUN METADATA
  seed = 0

WARNINGS — these must be stated in the report, not omitted
  [exec.no_swallowed_traceback] the experiment caught an exception and continued

FULL EXPERIMENT OUTPUT (14,208 bytes, untruncated at .../stdout.txt)
...
```

Only raw stdout is budgeted, and generously; no citable value lives there that is not already in
the registry above it.

---

## 5. Artifacts

Per attempt, under `<research_dir>/gate_artifacts/gate1/attempt_NN/`:

| File | Contents |
|---|---|
| `experiment.py` | exactly what ran |
| `stdout.txt` / `stderr.txt` | complete, untruncated, separated |
| `results.json` | declared metrics with call-site provenance, plus environment |
| `gate1_report.json` | verdict and every check, with evidence |

Plus one line per attempt in `divergence.jsonl`, pairing the gate verdict with the reward score.

---

## 6. Verification

The rewritten execution path was checked against all four original defects:

| Original defect | Result |
|---|---|
| crash after >1000 chars reported as success | detected; marker at index 0 of the return |
| namespace leaked between runs | `NameError`, as it should be |
| output truncated to 1000 chars | 5,001 chars returned intact |
| timeout could not kill the thread | process group killed at 5.0s |

Against the archived run, stated precisely — the honest version is narrower than the tempting
one. The **crashed attempts** in the log are rejected statically by `static.no_unbound_names`.
The **saved** `run_experiments.py` is statically clean, because it is the last successful
`REPLACE` rather than the crashed one; Gate 1 rejects it on `results.contract_present` instead —
zero `record_result` calls, so not one number in that paper is citable. Different checks fire on
different artifacts of the same run. The claim to make is *"Gate 1 rejects this run"*, not
*"Gate 1's static tier catches this file"*.

---

## 7. Portability

`gates/` imports nothing from any host scaffold. Porting means writing one adapter next to
`gates/adapters/agentlab.py`, responsible for four things: build the config and context, call
`gated_execute` where the scaffold called its `exec` helper, charge one rewrite per agent turn,
and hand `render_feedback(report)` back on rejection.

The package has **zero runtime dependencies** — stdlib only — so it installs into whatever
environment the host already has.

---

## 8. Not yet done

Gate 2 (source ↔ result coherence) and Gate 3 (report validity) are specified in
[`PLAN.md`](PLAN.md) and not implemented. Gate 1 does not check whether results are *plausible*
given the literature, only whether they were *measured*.
