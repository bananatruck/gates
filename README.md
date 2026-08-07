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
| **1 — execution validity** | Did this code actually run, and were the reported numbers produced by *this* run? | implemented |
| **2 — source ↔ result coherence** | Are the measured results consistent with what the cited literature reports? | designed |
| **3 — report validity** | Does every number and citation in the manuscript trace to something that exists? | designed |

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
than the result of a computation.

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
- `exec.no_swallowed_traceback` *(warns)* — the experiment caught an error and
  carried on

Results contract:

- `results.contract_present`, `results.expected_keys_present`
- `results.values_computed` — no metric value is a source literal
- `results.values_finite`
- `results.non_degenerate` *(warns)* — exact zeros, perfect scores, chance-level
  accuracy. This is the limitation AutoResearchClaw reports for value
  registries: the zeros are real measurements, so we surface them rather than
  accept them silently.

Failures produce a small, specific report for the agent, and
`gate1_report.json` for the record.

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
results.json        the declared metrics, with call-site provenance
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

## License

MIT.
