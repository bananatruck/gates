# G.A.T.E.S.

A portable validity layer for autonomous research agents.

Autonomous research scaffolds — Agent Laboratory, AI-Scientist-v2, and their
descendants — regularly publish numbers their experiments never produced. The
usual explanation is that the model fabricates. In the scaffold we audited, the
mechanism was mundane and mechanical: experiment output was truncated to 1,000
characters before any agent could read it, and the crash marker was appended
*after* the program's own output, so it fell off the end of the same slice. The
writing agent could not see the real numbers, and the failure detector could not
see the crash. A run that raised `NameError` on every attempt was scored `1.0`
by the reward model and written up with a full results table to two decimals.

G.A.T.E.S. treats that as an information-flow defect rather than a model
tendency, and closes the channel at three points.

| Gate | Question | Status |
|---|---|---|
| **1 — execution validity** | Did this code actually run, and were the reported numbers produced by *this* run? | **complete and measured** |
| **2 — source ↔ result coherence** | Are the measured results consistent with what the cited literature reports? | designed |
| **3 — report validity** | Does every number and citation in the manuscript trace to something that exists? | designed |

Gate 1 is finished for the scope it declares. It has been run against a complete
controlled A/B campaign on a real model and a real scaffold; the numbers are in
[Measured results](#measured-results), and the boundary it does not cross is in
[What Gate 1 does not do](#what-gate-1-does-not-do).

Full design: [`docs/PLAN.md`](docs/PLAN.md).

---

## Install

```bash
pip install -e /path/to/gates
```

Zero runtime dependencies — stdlib only, so it drops into whatever environment
the host scaffold already has.

## Use

```python
from gates import Gate1Config, run_gate1, render_feedback

report = run_gate1(experiment_source, Gate1Config(
    expected_keys=("exp1.K2.test_acc", "exp2.speedup"),
    timeout_s=900,
))

if not report.passed:
    send_back_to_the_engineer(render_feedback(report))
else:
    verified = {k: m.value for k, m in report.metrics().items()}
```

Experiments declare results through an API injected into their namespace — no
import, nothing for the agent to get wrong:

```python
record_result("exp1.K2.test_acc", test_acc, unit="ratio")
```

Pass the *variable*. A number typed into the call is rejected: Gate 1 re-parses
the source, finds the call site, and fails any value that is a literal rather
than the result of a computation. Assigning the literal to a name first does not
launder it — the same pass follows each name back through its bindings and
reports a value whose every input is a constant.

## What Gate 1 checks

Static, before anything runs — so a broken program costs no compute:

- `static.syntax_valid` — compiles
- `static.no_unbound_names` — no name read but never bound (uses `symtable`, so
  closures, comprehensions, `global`, walrus and class scopes are handled by the
  same machinery the interpreter uses)
- `static.no_banned_calls` — no `exit()`/`sys.exit()` forging a clean exit code

Runtime, in a fresh process with an empty namespace:

- `exec.exit_code_zero`, `exec.no_uncaught_exception`, `exec.completed_within_budget`
- `env.clean_namespace` — nothing inherited from a previous attempt
- `env.code_identity` — the source the child process ran hashes to the source
  submitted, which is what makes "this value came from this code" checkable
  rather than assumed
- `exec.no_swallowed_traceback` *(warns)* — the experiment caught an error and
  carried on. Scans **both** streams: `except Exception as e: print(e)` puts the
  evidence on stdout, where a stderr-only check never sees it
- `logs.no_error_signals` *(warns)* — trouble the run reported and continued
  past: numerical warnings that silently corrupt a metric (`invalid value
  encountered`, `divide by zero`), CUDA failures and CPU fallbacks,
  non-convergence, `ERROR`-level logging. Tuned for precision over recall, since
  a false positive costs the agent a rewrite for nothing — `ValueError:` is
  flagged, `Mean Squared Error:` is not
- `env.seed_recorded` *(warns)* — a run with no declared seed cannot be
  re-executed to check its own numbers

Results contract:

- `results.contract_present`, `results.expected_keys_present`
- `results.values_computed` — no metric value is a source literal
- `results.values_traced` *(warns)* — no metric value resolves to source
  literals once its variables are followed back to their bindings.
  `results.values_computed` reads the call site alone, so `record_result("k",
  0.816)` fails it and `acc = 0.816` followed by `record_result("k", acc)` does
  not — one indirection is the only difference. This check follows the names.
  It warns rather than fails because a genuinely constant value (a configured
  batch size recorded beside the metrics) is indistinguishable from a fabricated
  one without knowing what the number means, which is Gate 2's question
- `results.values_finite`
- `results.declared_keys_only` *(warns)* — keys recorded that the plan never
  declared. `results.expected_keys_present` tests presence, not equality, so an
  experiment can satisfy its contract while recording anything else. That
  matters beyond tidiness: when a scaffold prepends an earlier phase's code to
  this one, the earlier phase's keys arrive here and can satisfy a contract this
  run never met
- `results.single_observation` *(warns)* — a key recorded repeatedly with a
  changing value. Every call is retained, so a metric written once per epoch
  arrives with its whole trajectory and the report has to say whether it means
  the final value or the best one
- `results.non_degenerate` *(warns)* — exact zeros, perfect scores, chance-level
  accuracy. This is the limitation AutoResearchClaw reports for value
  registries: the zeros are real measurements, so we surface them rather than
  accept them silently.

Failures produce a small, specific report for the agent, and
`gate1_report.json` for the record.

Requirement-by-requirement traceability, with each check tied to the test that
holds it in place, is in [`docs/GATE1_REQUIREMENTS.md`](docs/GATE1_REQUIREMENTS.md).

## The value registry

Gate 1's output artifact — and the only thing a manuscript may cite from. Each
value carries its type, its unit, a `trace_id` binding it to one execution, and
its provenance chain, link by link:

```
task  →  command  →  log  →  value        (Gate 3 appends: claim)
```

Each link is marked resolved or not, so an unresolved chain is *reported* rather
than papered over — and `chain_integrity` gives the rate across all values. That
distinction is the point: a number that merely matches something in a log is not
the same as a number causally attributable to a recorded run, and conflating them
is what lets a fabricating agent satisfy a checker by keeping its numbers merely
consistent.

Every rejected run still gets a registry, recorded as `"citable": false`, so a
consumer that forgets to check the verdict still cannot cite it. That holds for
pre-execution rejections too: a program rejected on a syntax error has nothing
to record, so its registry is empty — but the file exists and says it is not
citable, rather than being absent and inviting a consumer to read the absence as
an unrelated error.

## Measured results

One controlled A/B campaign, both arms driven by `deepseek-v4-flash` from a single
config whose SHA-256 is identical on each side (`a0409cf…3de0e`), so the only
difference between the arms is whether Gate 1 is in the loop. Full method and
evidence index: [`reports/finalized-report-and-results/`](reports/finalized-report-and-results/).

### The evidence channel

The headline is a same-artifact result: given the *identical* execution, how much
of it reaches the next phase?

| Measure | Gate 1 on | Gate 1 off |
|---|---:|---:|
| Required key/value pairs delivered downstream | **40/40** | **0/40** |
| Attempts delivering all four required values | **10/10** | **0/10** |
| Citable registry values in the final run | 4 | 0 |
| Required values reported numerically in the paper | **4/4** | **0/4** |

The 1,000-character channel delivered none of the 40 required pairs it had
already recorded. That is the defect stated as a number: the measurements existed
and could not be read.

### Traceability of the generated paper

| Measure | Gate 1 on | Gate 1 off |
|---|---:|---:|
| Claims sourced directly from the registry | 19/29 (65.5%) | 0/11 (0%) |
| Claims registry-sourced **or** checkably derived | **28/29 (96.6%)** | **0/11 (0%)** |

### The crash the upstream detector cannot see

Reconstructed exactly, because it is a substring search over a fixed slice rather
than a model: for the archived crash shape the `[CODE EXECUTION ERROR]` marker
lands at character **3,442**, so a 1,000-character channel could not have seen it
and a 4,000-character one could. Across the rig's scenarios, Gate 1 rejected 6
engineer turns of which upstream's detector would have accepted 3.

### The log scanner

Against `tests/fixtures/log_corpus.jsonl`, 68 lines, 34 positive signals, Wilson
95% intervals:

| Scanner | Precision | Recall |
|---|---|---|
| deterministic patterns | 1.000 (0.824–1.000) | 0.529 (0.367–0.685) |
| + `qwen3:8b`, 3-shot | 1.000 (0.887–1.000) | 0.882 (0.734–0.953) |

Precision is the floor and the corpus enforces it; recall is what the model layer
exists to move. Model findings are WARN-only **by construction** and cannot reach
the verdict.

### The constant-taint pass, against real agent output

`results.values_traced` was run over all 20 experiment sources the campaign
produced — 203 `record_result` call sites in total. Every one classified as
`computed`: **zero false positives on real agent code**. The check is designed to
under-report, since a false warning costs the engineer a rewrite for nothing.

### What did *not* improve

Stated because it is the honest boundary and it bounds the claim. Mean of three
final reviewer scores: **3.735 gated vs 3.765 ungated** — essentially equal, and
both papers were recommended for rejection. Gate 1 delivers the evidence channel.
It does not make the science good, and no number here should be read as saying it
does.

## What Gate 1 does not do

Gate 1 answers one question — *did this code run, and did these numbers come from
this run?* — and these are outside it by design rather than by omission:

- **`results.values_computed` is a call-site literal check, not proof of
  measurement.** `results.values_traced` extends it through variable bindings,
  but a constant that passes through anything the pass cannot evaluate — a
  function call, a loop, a container it reads back out — is reported as computed.
  Neither check is an anti-fabrication proof, and neither should be described as
  one.
- **Whether a value is scientifically meaningful is Gate 2's question.** Gate 1
  will pass a correctly measured number that means nothing.
- **Whether the manuscript's prose follows from the numbers is Gate 3's.** In the
  campaign, the gated writer still derived a scaling exponent from two points and
  described a single seeded dataset as free of sampling variance. Gate 1 passed
  the run that produced them, correctly.
- **Task compliance is not checked.** The gated code imported a fixture despite a
  NumPy-only instruction. It executed, so Gate 1 passed it.
- **Static name analysis is conservative.** Module-level use-before-assignment is
  left to the runtime tier rather than guessed at statically.
- **Log scanning is bounded** at 2,000,000 characters per stream. The full
  capture stays on disk, untruncated.
- **This is process isolation, not a security sandbox.** The child runs as the
  same OS user. Provider credentials are scrubbed from its environment and the
  parent is marked non-dumpable on Linux (so `/proc/<ppid>/environ` and
  ptrace are closed), but production use should execute experiments in a
  container or a separate unprivileged account.
- **The A/B sample is one workflow per arm.** The ten candidates within a workflow
  are dependent optimization steps, not ten independent research tasks. The
  channel result is a same-artifact comparison and does not need n; any claim
  about *rates across tasks* does, and is not made here.
- **No MLR-Bench task was run**, and no cost claim is available — the scaffold
  printed a `$0.0` placeholder rather than measuring provider billing.

## Porting to another scaffold

`gates/` imports nothing from any host system. Porting means writing one adapter
next to `gates/adapters/agentlab.py`, which is the reference implementation
against Agent Laboratory's MLE solver. An adapter is responsible for:

1. building a `Gate1Config` and a `GateContext` for the phase,
2. calling `gated_execute` where the scaffold used to call its `exec` helper,
3. charging one rewrite per *agent turn* (not per execution — automated repair
   loops must not eat the agent's budget),
4. handing `render_feedback(report)` back to the agent on rejection.

## Artifacts

Every attempt writes to `<artifact_root>/gate1/attempt_NN/`:

```
experiment.py       exactly what ran
stdout.txt          complete, untruncated
stderr.txt          complete, untruncated
results.json        the declared metrics, with call-site provenance and the
                    full observation history for every key
registry.json       the citable value registry: typed values, trace ids,
                    provenance chains, chain-integrity rate
gate1_report.json   the verdict and every check
```

and one line per attempt to `divergence.jsonl`, pairing the gate's verdict with
what the reward model scored the same attempt — against the 1,000-character view
the upstream scaffold would have had. That file is the evidence table, not a
debugging aid.

## Tests

```bash
pip install -e ".[dev]" && pytest
```

302 tests here, plus 76 in the host scaffold's integration suite — 378 in total,
all green. Every check in the tables above is tied to the test that holds it in
place in [`docs/GATE1_REQUIREMENTS.md`](docs/GATE1_REQUIREMENTS.md).

## Seeing the loop run

The checks are testable one at a time; the thing they compose into is a loop, and
`rig/` drives that loop end to end with a scripted engineer standing in for the
model. Every code path is the shipped one — only the source of the code is fake,
which is what makes it free to run on every change.

```bash
python -m rig.gate1_loop              # every scenario, full transcript
python -m rig.gate1_loop archived-run # replay the audited run: 3 turns, no paper
python -m rig.gate1_loop --quiet      # summary table only
```

Five scenarios cover the failure modes Gate 1 claims: the audited run replayed,
a rejection the agent answers by inventing numbers, three executions inside one
engineer turn, a run that passes with four warnings, and a namespace leak. Each
declares the checks it must fail, the turns it must consume and its terminal
state, so a scenario cannot drift from its own description. Exit status is
non-zero on any mismatch.

The rig also reconstructs the host scaffold's own failure detector — a substring
search over a 1,000-character slice, so it reproduces exactly — and reports which
attempts Gate 1 rejected that upstream would have accepted. It does **not**
reconstruct `get_score`; that is an LLM at temperature 0.6, so the ledger's
reward column stays `null` unless a real model is passed to `run_loop`.

What was outstanding for Gate 1 and how each item closed:
[`docs/GATE1_COMPLETION.md`](docs/GATE1_COMPLETION.md). Nothing there blocks
Gate 1; the one open item is a second adapter, which is a portability
demonstration rather than a Gate 1 requirement.

## License

MIT.
