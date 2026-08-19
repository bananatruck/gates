# Gate 1 — requirements traceability

Every requirement Gate 1 is answerable for, traced to the check that satisfies it
and the test that holds it in place. Three sources:

| Tag | Source |
|---|---|
| **V** | the project's spoken specification of Gate 1 |
| **D** | `verification-layer-proposal.pptx` — slide numbers as cited |
| **P** | `26_Keshav.pdf` — *ARGUS: Adversarial Verification Layer*, sections as cited |

Status is one of **MET**, **PARTIAL** (satisfied for the part inside Gate 1's
remit, with the remainder named), or **OUT OF SCOPE** (belongs to Gate 2 or 3, and
is recorded here so it is not mistaken for an omission).

Tests live in `tests/test_gate1.py` unless noted.

---

## The spoken specification

| # | Requirement | Satisfied by | Test | Status |
|---|---|---|---|---|
| V1 | The generated code "has very real variables" | `static.no_unbound_names`, `env.clean_namespace`, `results.values_computed` | `test_unbound_name_inside_method_is_found`, `test_namespace_does_not_leak_between_runs`, `test_hardcoded_value_is_rejected` | **MET** |
| V2 | Executable, "with no runs failed" | `exec.exit_code_zero`, `exec.no_uncaught_exception`, `exec.completed_within_budget` | `test_crash_after_heavy_output_is_caught`, `test_timeout_kills_the_run` | **MET** |
| V3 | "log files without errors that don't seem really code-breaking" | `exec.no_swallowed_traceback` and `logs.no_error_signals`, both WARN, against the FAIL tier above | `test_traceback_printed_to_stdout_is_caught`, `test_numerical_warning_is_surfaced`, `test_real_error_signals_are_flagged`, `test_ordinary_ml_logging_is_not_flagged` | **MET** |
| V4 | Sits between execution and the reward system | `adapters/agentlab.py` — `gated_execute` runs before `score_and_log`; the solver's success test is the verdict, not a marker string | `tests/test_gates_integration.py` (Agent-Researcher) | **MET** |
| V5 | The reward system does not decide validity | Gate 1 consults no model. `get_score` ranks passing attempts only; disagreements land in `divergence.jsonl` | `test_ledger_records_divergence` | **MET** |
| V6 | On failure, a feedback loop to the ML engineer with a small report | `render_feedback`; budget counts agent rewrites, not executions | `test_feedback_report_names_the_line_and_the_fix`, integration tests | **MET** |

**On V3.** The distinction the spec asks for is between errors that break the run
and errors the run continued past. The first tier already hard-fails. The second
is what `logs.no_error_signals` covers: numerical warnings that silently corrupt a
metric (`invalid value encountered`, `divide by zero`), device failures and CPU
fallbacks, non-convergence, printed exceptions, and `ERROR`-level logging. Both
streams are scanned — an agent that writes `except Exception as e: print(e)` puts
the evidence on stdout, where a stderr-only check never sees it.

These never block, because blocking on them would reject legitimate runs. They are
routed into the feedback report and into the evidence bundle the writer receives,
under a heading that says they must be stated rather than omitted.

Pattern precision is a deliberate design constraint: a false positive costs the
agent a rewrite for nothing. `Mean Squared Error:`, `Standard Error:`, `test error
rate:` and `Converged after 40 iterations` are all held un-flagged by test.

---

## The proposal deck

| # | Requirement | Source | Satisfied by | Test | Status |
|---|---|---|---|---|---|
| D1 | Untruncated stdout plus a typed `results.json` written by the experiment itself | slide 11, step 1 | `runner.run_experiment` captures to files, never `[:MAX_LEN]`; `harness.py` writes the typed record; `output.untruncated` records the sizes | `test_crash_after_heavy_output_is_caught`, `test_artifacts_are_written` | **MET** |
| D2 | Uncaught exceptions, non-zero exit codes and missing expected keys hard-fail the phase | slide 11, Gate 1 | the three FAIL checks of those names | `test_missing_declared_key_is_rejected` and V2's tests | **MET** |
| D3 | A verified value registry: every number the paper may use, **typed and hashed to the run that produced it** | slide 11, step 3 | `registry.json` — per-value `trace_id`, `unit`, `type`, and `run.code_sha256_verified`, cross-checked by `env.code_identity` | `test_registry_binds_each_value_to_the_run`, `test_executed_source_is_hashed_by_the_process_that_ran_it`, `test_trace_ids_differ_between_runs_of_identical_code` | **MET** |
| D4 | Silent failure scored as success — eliminated by construction; a crashed run cannot reach the writing phase | slide 12 | `GateContext.check_can_continue` raises `GateFailure` when the budget is spent and nothing passed; otherwise the last passing attempt is used | `test_gate_failure_message_names_the_checks`, integration tests | **MET** |
| D5 | No model is asked whether the run succeeded | slide 7 | `gate1.py` imports no model client; every verdict is a runtime fact | — (structural) | **MET** |
| D6 | AutoResearchClaw's reported limitation: a registry passes zero-valued results because the zeros are real | slide 8 | `results.non_degenerate` surfaces exact zeros, perfect scores and chance-level values as WARN, never blocking | `test_degenerate_values_warn_only` | **MET** |
| D7 | Replace `get_score()` | slide 11 | Demoted rather than deleted, by decision: Gate 1 owns pass/fail, `get_score` orders passing attempts, and every disagreement is logged as the paper's evidence | `test_ledger_records_divergence` | **MET, by amendment** |
| D8 | Fabrication rate as a function of channel fidelity — the novel experiment | slide 10, slide 15 step 3 | `Gate1Config(require_metrics=False)` and `GatedExecution.legacy_view()` provide the degraded arm and the faithful 1000-character reconstruction | `test_contract_optional_in_ablation_mode` | **MET** — run; 40/40 required pairs delivered gated, 0/40 ungated, same artifacts |
| D9 | Metrics-bearing outputs, numeric fabrication rate, CORE-Bench accuracy | slide 13 | evaluation targets, not implementation | — | **PARTIAL** — paper traceability measured (28/29 gated vs 0/11 ungated); no CORE-Bench or MLR-Bench submission was run, and none is claimed |

---

## The ARGUS proposal

| # | Requirement | Source | Satisfied by | Test | Status |
|---|---|---|---|---|---|
| P1 | Causal provenance, not value-matching: an unbroken chain, prompt → command → log → value → claim | Sub-Topic C | `registry.json` carries `chain` per value, link by link, each marked resolved or not. Gate 1 establishes `task` (from `Gate1Config.task_ref`), `command` (run id + argv + verified hash), `log` (the capture on disk) and `value` (the trace id). Gate 3 appends `claim`. | `test_registry_binds_each_value_to_the_run`, `test_chain_reports_the_missing_link_when_no_task_is_supplied` | **MET for its four links** |
| P2 | A value that matches a log but has no recorded command that produced that run is a worse signal than a numeric discrepancy | Sub-Topic C | An unresolved link is recorded rather than papered over; `chain_complete` is false and `missing_links` names which. A rejected run's registry reports `citable: false`, so its values cannot be cited even by a consumer that ignores the verdict. | `test_rejected_run_yields_no_citable_values`, `test_chain_reports_the_missing_link_when_no_task_is_supplied` | **MET** |
| P3 | Two-tier release-blocking gate: Block tier vs Flag tier | Sub-Topic E | `Severity.FAIL` blocks; `Severity.WARN` flags and propagates into the report; `decide()` implements the split | `test_degenerate_values_warn_only`, `test_missing_seed_warns_but_does_not_block` | **MET** |
| P4 | Native failure mode: silently reporting a best-epoch number instead of the final-epoch number | Sub-Topic B | `record_result` retains every call, not just the last. `results.single_observation` warns when a key's value changed between calls and reports the span, so the report must say which value it means. | `test_repeated_record_keeps_every_observation`, `test_observation_history_is_capped` | **MET** |
| P5 | Native failure mode: dropping a failed run from an averaged metric without disclosure | Sub-Topic B | The observation history and `call_count` expose the drop **when each replicate is recorded**. A mean computed inside the experiment over a silently shortened list is one recorded value and Gate 1 cannot see behind it. | `test_repeated_record_keeps_every_observation` | **PARTIAL** — a replicate contract (`record_result` with per-seed values, per AutoResearchClaw) is the remaining piece; it is Gate 2's, since deciding whether a mean is honest needs the plan's declared seed count |
| P6 | Perturbation robustness: truncated compute budgets, corrupted or partial logs, seed/run interruptions | Sub-Topic B, Phase I | `exec.completed_within_budget` kills the process group and reports it as a timeout, not as a missing contract; an unreadable `results.json` becomes `harness_error` and fails `results.contract_present`; `env.seed_recorded` warns when a run cannot be reproduced | `test_timeout_kills_the_run`, `test_timeout_message_does_not_blame_the_contract`, `test_missing_seed_warns_but_does_not_block` | **MET** |
| P7 | Causal-chain integrity rate as a reported metric | §4 New Metrics | `registry["chain_integrity"]` — complete chains over recorded values, with the missing links per key | `test_registry_binds_each_value_to_the_run`, `test_chain_reports_the_missing_link_when_no_task_is_supplied` | **MET** |
| P8 | An A2 substrate: a structured metadata file proving causal provenance, not just a value match | §5 Expected Outcomes | `registry.json`, one per attempt, plus `gate1_report.json` and `divergence.jsonl` | `test_artifacts_are_written` | **MET** |
| P9 | A Trace ID per claim, pointing back to specific run logs | Topic 2, Sub-Topic B | `trace_id = sha256(run_id, key, lineno)`; `registry.run.stdout_path` and `results_json_path` are the log it points at; `resolve_trace()` performs the lookup | `test_registry_binds_each_value_to_the_run` | **MET** |
| P10 | Execution-grounded evaluation — re-running the code — as ground truth | Topic 2, §5 | Made possible rather than performed: `env.code_identity` fixes exactly what to re-run, `env.seed_recorded` warns when re-running cannot reproduce, and `registry.run.argv` records the command | `test_executed_source_is_hashed_by_the_process_that_ran_it` | **MET as a precondition** |
| P11 | Native failure mode: describing a plot never regenerated after a late code change | Sub-Topic B | — | — | **OUT OF SCOPE** — Gate 3. Gate 1 makes it detectable by binding figures to the run that produced them (`ai_lab_repo` regenerates figures through `gated_execute`) |
| P12 | Fine-grained claim extraction; adversarial Fabricator/Auditor co-evolution; VLM plot reading | Sub-Topics A, D; §3 Phase III | — | — | **OUT OF SCOPE** — Gate 3 and the evaluation programme. Gate 1 supplies the substrate they audit against |

---

## Limitations closed after validation

The validation campaign's audit produced a list of gaps. Three were cheap and are
now closed in code rather than carried as prose; each is held by a test.

| # | Gap | Satisfied by | Test | Status |
|---|---|---|---|---|
| L1 | `values_computed` reads the call site only, so a literal assigned to a variable first satisfies it | `results.values_traced` — resolves each name back through its bindings, WARN by construction | `test_value_laundered_through_a_variable_warns_but_does_not_block`, `test_a_real_measurement_is_not_called_constant`, `test_a_name_bound_outside_plain_assignment_is_not_called_constant` | **MET** |
| L2 | Expected keys are presence-only, so an earlier phase's keys can satisfy this phase's contract | `results.declared_keys_only` — reports undeclared keys, WARN by construction | `test_keys_the_plan_never_declared_are_reported`, `test_exactly_the_declared_keys_raises_nothing` | **MET** |
| L3 | Documentation claimed every rejected run has a registry; pre-execution rejections had none | `write_registry` runs on the static-rejection path, writing an empty registry marked `"citable": false` | `test_a_statically_rejected_run_still_gets_a_registry` | **MET** |

Both new checks warn and neither can change a verdict — the verdict stays
deterministic and stays where `decide()` puts it. `results.values_traced` was run
over all 20 experiment sources the campaign produced, 203 `record_result` call
sites, and raised nothing: it is built to under-report, because a false warning
costs the engineer a rewrite for nothing.

## What Gate 1 does not claim

Stated plainly, because the deck's slide 12 commits to saying it precisely:

- **It does not check whether a result is plausible.** A correctly measured,
  correctly recorded, causally traced number can still be scientifically wrong.
  Comparing results against the cited literature is Gate 2.
- **It does not check the manuscript.** Nothing here reads LaTeX. Whether a claim
  in prose is supported by a registry value is Gate 3.
- **`static.no_unbound_names` is conservative on purpose.** Module-level
  use-before-assignment is not reported, because deciding it needs ordering
  analysis and would produce false positives. The runtime tier catches what the
  static tier declines to claim, one execution later.
- **`logs.no_error_signals` has recall it cannot bound.** The pattern set covers
  the signals observed in the archived runs and the common numerical and device
  failures. An error message outside it passes silently. Precision was preferred
  to recall, and that is a measured trade, not an oversight.
- **`results.values_traced` is a heuristic, not a proof.** It follows plain
  assignments and constant-folding conversions. A constant that passes through a
  function call, a loop, or a container it is read back out of is reported as
  computed. It closes one indirection, which is the one observed; it does not
  make either value check an anti-fabrication proof, and neither should be
  described as one.
- **P5 is partial** and is marked so above rather than rounded up.
