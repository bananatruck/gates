# progress

Read this before starting a session. Update it when a task **completes**, not when it starts.
High-density only: SHAs, verified counts, decisions, blockers, exact next steps. No speculation, no narrative.

The `STATE` block below is machine-checked by `tests/test_progress.py`. If you change the tree
without updating it, the suite fails. That is the point — every other status doc in this repo has
gone stale at least once.

**Not machine-checked:** `branch`, `head`, `head_date`. A commit cannot record its own SHA, so a
checked `head` would be unsatisfiable by construction. Only the counts are verified, and claiming
otherwise would be the same overclaim as a green check that never ran. Update these three by hand.

## state

<!-- STATE:BEGIN -->
branch: main
head: dc31079
head_date: 2026-09-16
tests_total: 625
tests_gate1: 98
tests_gate2: 92
tests_gate3: 84
tests_llm_scan: 21
tests_llm_layer: 14
<!-- STATE:END -->

Named suites above do not cover every test file; `tests_total` is checked against the whole tree,
so a new test file surfaces as a total mismatch even without its own key.

Known environment-dependent result: `test_experiment_child_cannot_read_parent_proc_environment`
**skips** on macOS (`Linux /proc test`) and fails as root on Linux (uid 0 bypasses the
`PR_SET_DUMPABLE` guard via `CAP_SYS_PTRACE`). CI runs non-root Linux so CI is green. See B3.

## architecture

Decided 2026-09-10/11, amended 09-16 (D29-D31). **Tiers are a Gate 2 mechanism only.** Gate 3 mirrors Gate 1 (D30):
a flat list of checks with severities, no tier structure, a reject-and-retry loop back to the agent that
wrote the manuscript, a raise on a spent budget, and Gate 1's model layer rebuilt for manuscripts (D31).
The loop code is `report_loop` in the adapter, shaped like `review_loop` (D29).

**Gate 2 — three tiers**

| Tier | What it does | Input required |
|---|---|---|
| A | boundaries — deterministic ranges, declared relations, declared plausibility | none for ranges; a speedup for plausibility |
| B | methodology conformance — declared plan vs recorded run | `plan_fields` |
| C | harness / feedback loop | wraps A+B |

**Gate 3 — flat check list, Gate 1's shape**

| Family | What it does |
|---|---|
| `report.*` | numeric and figure binding (built) |
| `source.*` | citations checked against the retrieval registry and an online database |
| `style.*` | deterministic report-format checks before publishing |

Families are for discussion, not tiers. No ordering, no separate severity policy, no conditional
activation beyond the usual rule: a check with no input emits nothing.

**Deliberate asymmetry, state it in the paper:** on budget exhaustion Gate 2 proceeds with the
discrepancy declared; Gate 3 raises and emits nothing. A genuine novel result must not be blocked
forever, but an unverifiable manuscript must not ship.

## components

| Component | State | Entry point | Tests |
|---|---|---|---|
| Gate 1 — execution validity | complete, evidence frozen | `gated_execute()` | 98 |
| Gate 2 tier A — boundaries | **complete** — ranges, relations, non-finite guard, plausibility ceiling | `run_gate2()` | part of 92 |
| Gate 2 tier B — methodology conformance | **complete** — `method_conformance` FAIL, `method_traceable` WARN; host path via `make_review_context(plan_fields=, sources=, lit_review=)` | `run_gate2()` | part of 92 |
| Gate 2 tier C — loop | **complete, model-free**. 6 scenarios. Every revision runs under Gate 1 first (F12). A spent Gate 2 budget proceeds with the discrepancies declared. A spent Gate 1 budget raises if nothing ran clean. `first=` reviews a pass the host already holds. The outcome carries `registry` and `declared`. **Host calls it** from `running_experiments` after `gated_execute` (D42). | `gates/adapters/agentlab.py` `review_loop()`, driven by `rig/gate2_loop.py` | 26 |
| Gate 2 tier comparison | complete. A vs A+B vs A+B+C | `rig/gate2_tier_comparison.py` | 8 |
| Gate 3 — `report.*` checks | numeric + figure binding; `limitations_declared` (step 5, D35); D14 guard | `run_gate3()` | 30 of 57 |
| Gate 3 — `source.*` | **complete for this host.** `cited_papers_in_registry` (step 6: D21, D25, D26, D32) with `retrieved_arxiv_ids()`; `identifiers_resolve` (step 9, D41, D49) with `PaperRecord` in `gates/schema.py` and `arxiv_lookup()` in the adapter. `citations_parse` and `metadata_agrees` have no input here: inline citations, no bibliography | `run_gate3(retrieved=, )`, `Gate3Config.lookup` | 20 of 83, plus 11 in `tests/test_arxiv_lookup.py` |
| Gate 3 — `style.*` | **complete.** `claim_sections_bound` (step 4, D34); `sections_present` (7a, D27/D40), host list in `WRITER_SECTIONS`; `no_orphan_references` (7b, D33); `floats_referenced` (7c, D33, WARN). `acronyms_defined` dropped by D33 | `run_gate3()` | 25 of 75 |
| Gate 3 loop | **complete for the existing checks, model-free.** Adapter half (step 2): `make_report_context()`, `gated_report()`, `report_loop()`, `ReportOutcome`; a spent budget raises; a Gate 1-rejected registry is refused before `write` runs. Rig half (step 3): 7 scenarios, all six plan scenarios; the registry and declared limitations come from a real Gate 2 run (`clean` unless a scenario names another); a raised loop is rebuilt from `context.history`. | `gates/adapters/agentlab.py` `report_loop()`, driven by `rig/gate3_loop.py` | 9 of 57 in `tests/test_gate3.py`, 17 in `tests/test_gate3_loop.py` |
| LLM scan layer | complete, Gate 1 only | `gates/llm_scan.py` | 21 |
| Gate 3 model layer (D31) | complete, model-free by default. Claim scan (`report.model_unbound_claims`, WARN, quote-grounded, INFO when it cannot run); REQUIRED FIXES via `generate_fixes(system=, facts=, grounding=)` with `check_manuscript_grounding`; `llm_report.attach_fixes` shared with Gate 1; `make_report_context(consult_model=)`; `REPORT_GATE_INSTRUCTIONS` | `gates/llm_claims.py`, `gates/gate3.py` | 23, in `tests/test_gate3_model.py` |
| LLM plumbing (`ModelFn`, budget) | complete, Gates 1 and 3 | `gates/llm.py` | 14 |
| `gates/gate2_semantic.py` | **deleted** 09-13 (D4). Gate 2 is model-free (D19) | - | 15 removed |
| Agent Laboratory adapter | `gated_execute`; `gated_review` and `review_loop` via `make_review_context()`, which takes `relations`, `ranges`, `plan_fields`, `sources` + `lit_review`; `gated_report` and `report_loop` via `make_report_context()`. Host call sites on `AgentLaboratory-Gemini` `feat/gates-d42-connect`: `running_experiments` -> `review_loop(..., first=final)`, `report_writing` -> `report_loop` | `gates/adapters/agentlab.py` | host: `tests/test_gates_d42.py` |
| Tier A evaluation harness | complete | `rig/gate2_tier_a_eval.py` | 45 labelled registries |

Frozen and checksum-signed — do not edit: `reports/finalized-report-and-results/`.

## gate 2 - status

**Complete in this repo as of 09-16.** Tiers A, B and C are built and connected: `review_loop` runs code under Gate 1, then reviews the registry under A+B. The host call sites landed 09-16 on `AgentLaboratory-Gemini` `feat/gates-d42-connect` (D42). Remaining items need a live gated run, declared `plan_fields` (F2), or model spend.

Verified 09-14 against `85af14c` and `../AgentLaboratory-Gemini` (`feat/gates-verification-layer`, 7 files uncommitted).
**Closed 09-14:** F4 (`0baf625`); F1 + F3 at adapter level, `review_loop` + `ReviewOutcome.declared` (`0baf625`); F6, `make_review_context(sources=, lit_review=)` (D23, `12aa1d9`); F10, exact published tallies asserted in `tests/test_gate2.py` (`60af585`); F7, `Ledger.loop_summary()` from `review_loop` rows (M5; a run enters the loop only if its first review failed). F12 (09-16), `review_loop(..., gate1=)` runs every revision under Gate 1, so Gate 2 never reviews a registry no run wrote.
**Scope 09-14, amended D42:** gates repo plus host call sites on a branch. The live gated run is still budgeted.
Tiers A and B run as one call (`run_gate2`). Tier C is `review_loop` in the adapter, which the rig drives and the host now calls. The open items:

| ID | Missing | Evidence | Needs |
|---|---|---|---|
| F2 | Nobody declares `plan_fields` or `relations` in a real run, so tier B never activates and tier A checks ranges only. | D13; the host plan is free text | a declaration source |
| F5 | Setup budgets reach nothing. **Host side only**: both context builders take `max_attempts` and their defaults match `setup.defaults()`. | `gates/setup.py:121` prints JSON only | host passes the chosen budget (out of scope 09-14) |
| F8 | M6 wallclock overhead is not measured. | spec §4 M6 | paired timing, with F9 |
| F9 | E1 not run: MLR-Bench's 10 tasks, gated vs ungated, paired. | spec §5 E1 | F1, F2, model spend |
| ~~F11~~ | **Closed 09-16.** `SKILL.md` + `tests/test_install_skill.py`. | — | — |

## blockers

| ID | Blocker | Blocks | Opened |
|---|---|---|---|
| B3 | `PR_SET_DUMPABLE` guard is inert under uid 0; not detected, not reported. Skips on macOS. CI runs non-root so CI is green. | portability claim | 09-07 |

**Closed:** B2 (Gate 3 `source.*` needs network) - 09-16: `arxiv_lookup()` in the adapter, injected through `Gate3Config.lookup`, with a `.cache/` disk cache and 3 s spacing (D41, D49). The suite stays hermetic on a dict-backed fake; one live test runs only under `GATES_LIVE_ARXIV=1` and passed 09-16. B4 (Gate 3 headline) - 09-16: `ARCHIVED` reads `generated_report.txt`, 29 literals in abstract, results, discussion; was `generated_readme.md`, 8. B8 (decoy hole) - 09-14: Gate 1 records `provenance.used_by_run` from `static_checks.find_unused_record_values`; tier B reports a recorded-but-unread plan value as unverifiable `unused` (WARN, D17). Ceiling: names only, see the `ponytail:` comment. B1 (benchmark selection) — MLR-Bench chosen, see D8. B7 (Gate 2 host entry point) — `make_review_context()` added 09-11; `test_the_host_wiring_path_actually_reaches_gate_2` keeps it reachable. B5 (unpushed WIP loop) - restated 09-13: that loop targets the retired `sources`/`consult_model` API, so tier C is rebuilt from `main`; the old code survives only on the local branch `backup/local-main-f1b9fe2`. B6 (host retrieval registry) - yes: Agent Laboratory keeps `self.lit_review`, entries keyed by `arxiv_id` (`AgentLaboratory/agents.py:574,682`); see D21.

## next steps

1. Gate 3, in this order (revised 09-16 from `GATE3_implementation_plan.md` §6). Red test first each step, full suite, commit, push.
   1. ~~B4~~ done.
   2. ~~Adapter entry point~~ done. Was: `make_report_context()` + `gated_report()` in the adapter, shaped like `review_loop`: `revise` returns the next manuscript; a spent budget raises `GateFailure`. Inputs from Gate 2: `ReviewOutcome.registry` (values to bind) and `ReviewOutcome.declared` (D28). `rig/loop.py` is Gate 1-wired (`run_loop`, `rig/loop.py:198`) and stays unchanged.
   3. ~~`rig/gate3_loop.py` + scenarios 1, 2, 3, 6~~ done. Was (existing checks only), plan approved 09-16: `clean`, `typed-literal-fixed`, `unknown-token`, `budget-exhausts`; the registry comes from a real `run_gate2_loop` of Gate 2's `clean` scenario, not a hand-built dict.
   4. ~~`style.claim_sections_bound` + scenario 5~~ done. Was: plus the "tiers" wording in the `gates/gate3.py` docstring.
   5. ~~Declared-limitations check (D28)~~ done (D35).
   6. ~~Registry (D25, D26), `source.cited_papers_in_registry` + scenario 4~~ done. `PaperRecord` moved to step 9.
   7. ~~Remaining `style.*`~~ done (D27, D33, D40): `sections_present`, `no_orphan_references`, `floats_referenced`. `acronyms_defined` dropped.
   8. ~~G3-M4~~ done (D48): 34 of 49 detected, 15 missed, 6 false positives, scanner unchanged (D38). Labels reviewed and accepted 09-16, figure published in `PLAN.md` §5.2 and the README.
   9. ~~`PaperRecord` and `source.identifiers_resolve`~~ done (B2 closed, D41, D49). `citations_parse` and `metadata_agrees` not built: no input on Agent Laboratory, which cites inline and keeps no bibliography.
   10. ~~Docs~~ done: `PLAN.md` §5.1/§5.2 (D43), README, `CLAUDE.md` §6 (D44).
   11. ~~review~~ done (`46ddc85`). Merged to `main` at `dc31079`.
   **Gate 3 is code-complete.** G3-M4 labels reviewed and accepted (D37).
   Approved 09-16 (Q8-Q14): D31 model layer and the Q14 key-leak test done; next steps 6-9.
   Approved 09-16 (second round, D37-D45): all recommendations accepted as written.
2. ~~Connect~~ done on `AgentLaboratory-Gemini` `feat/gates-d42-connect`. Live gated run still budgeted (D42, D45).
3. `env.parent_proc_guard` INFO check (B3).
4. ~~F11~~ done (D45). F9 + F8 next, with model spend. F2 waits on declared `plan_fields`. F5 waits on the host passing the chosen budget.

## decision log

Append-only. One line each: date, decision, where it is enforced.

| Date | ID | Decision | Enforced by |
|---|---|---|---|
| — | D1 | Deterministic checks alone decide verdicts | `gates/gate2.py`, `gates/gate3.py` |
| — | D2 | A check with no input emits nothing — absent, never green | tier derivation from supplied args, not flags |
| — | D3 | `gates/` is stdlib-only at runtime | `.github/workflows/tests.yml` stdlib check |
| 09-10 | D4 | **`gate2_semantic.py` is deleted.** Tier C is the loop; semantic checks have no tier. | `test_no_model_can_reach_a_gate_2_verdict` |
| 09-10 | D5 | Gate 3 reuses Gate 1's structure and loop; only check contents differ | `rig/loop.py` reuse |
| 09-10 | D6 | Claim entailment (MiniCheck) deferred past this paper | scope |
| 09-10 | D7 | Tiers are Gate 2 only. Gate 3 has a flat check list, like Gate 1. | `gates/gate3.py` |
| 09-11 | D8 | **Benchmark is MLR-Bench** (arXiv 2505.19955). CORE-Bench rejected: only 17 of its 181 task questions have stochastic answers. | `GATE2_implementation_spec.md` |
| 09-11 | D9 | **Tier B is methodology conformance**, not literature tolerance. `SourceClaim`/`Band`/`band_for` stay in the tree, unwired. | `GATE2_implementation_spec.md` §2 |
| 09-11 | D10 | **The speedup ceiling is gated on provenance, not magnitude.** A speedup a declared `Relation` derives is exempt at any size; SAGE reports a real 4,700x. Separate check `coherence.plausibility`, so `range_valid` stays provable. | `gates/gate2.py` `_check_plausibility` |
| 09-11 | D11 | **`IMPLAUSIBLE_SPEEDUP = 500.0`**, not 1000, to keep it clear of `MAX_LEN = 1000`, the stdout truncation this project diagnosed. Declared, not derived; reaches the report as `ceiling_origin`. | `gates/gate2.py` |
| 09-11 | D12 | **Bounded scores are units, never metric names.** `auc`/`f1`/`precision`/`recall`/`perplexity` in `UNIT_RANGES`. Name-based inference stays refused. | `test_a_metric_name_alone_never_implies_a_range` |
| 09-11 | D13 | **Tier B declarations arrive at wiring time**, not by parsing plan prose. The host passes `plan_fields`; `gates/` never reads a plan. Agent Laboratory's plan is free text (`ai_lab_repo.py:434`), so nothing is extracted. | `test_a_plan_declared_at_wiring_time_reaches_tier_b` |
| 09-12 | D18 | **A value recorded as a call-site literal cannot prove conformance.** `arg_kind == "literal"` means the number was typed at `record_result`, so matching it proves the agent typed it twice. `constant` (read from a binding) does prove it. | `test_a_value_typed_at_the_call_site_cannot_prove_conformance` |
| 09-12 | D17 | **Tier B is two checks, not one.** `method_conformance` FAIL for divergence, `method_traceable` WARN for unverifiable. Different findings, different remedies, and collapsing them would let "nobody can tell" read as "the run did something else". No `strict_conformance` flag: divergence is provable, so it fails, and Gate 2 proceeds on exhaustion anyway. | `gates/gate2.py` |
| 09-12 | D16 | **`PlanField` has no per-field tolerance.** A plan field is a declaration, not a measurement. A tolerance knob would let a run declare 0.001, use 0.0015, and widen until it conformed. Floats compare with representation slack only. | `test_float_slack_absorbs_representation_and_nothing_else` |
| 09-11 | D15 | **Gate 2 gets its own context builder.** `make_review_context()` builds a `Gate2Config`; `make_context()` stays Gate 1's. Two gates, two phases, two budgets, so one context holding both would need two rejection counters. | `test_the_host_wiring_path_actually_reaches_gate_2` |
| 09-11 | D14 | **Every check id must have an evidence renderer and a fix directive.** A keyed lookup that misses returns nothing and the agent gets a rejection it cannot act on. | `test_every_check_gate2_emits_can_be_rendered_and_has_a_fix` |
| 09-13 | D19 | **Gate 2 is model-free by construction.** Nothing `gate2.py` imports reaches `gates.llm`, and `Gate2Config` has no model field. Replaces `test_the_model_is_never_shown_a_value_the_registry_lacks`: no value can be shown to a model Gate 2 cannot call. | `test_no_model_can_reach_a_gate_2_verdict` |
| 09-13 | D20 | **No random-baseline plausibility band.** The spec's `suspicious_baseline` fixture (binary baseline at 0.65) is dropped: D12 refuses a range keyed on a metric name, and a fair coin on a small test set can land at 0.65, so the bound depends on sample size and is not a fact about the number. The other six E2 fixtures exist as cases in `rig/gate2_tier_a_eval.py` and `rig/gate2_tier_b_eval.py`. | scope |
| 09-14 | D23 | **Reference sources are declared, and bound to `lit_review`.** Partly reopens D9: `reference_interval` gets a host path. `SourceClaim`s arrive at wiring time like plan fields (D13); a `source_id` not in the host's `lit_review` raises `GateError`. Rejected: a model extracting claims from `full_text`, which would put a model upstream of a Gate 2 check (D19). Decided by Kesh. | `test_a_source_nobody_fetched_is_refused_at_wiring_time` |
| 09-13 | D22 | **Scenario 4 is an unverifiable field, not a justified divergence.** D17 made divergence FAIL, so a justification changes nothing and that story is scenario 5. Gate 2's only WARN-and-proceed path is a declared field nobody can check. | `test_an_unverifiable_plan_warns_proceeds_and_reaches_the_writer` |
| 09-16 | D24 | **Gate 2's loop takes code, not registries.** `review_loop` runs every revision under Gate 1 and reviews only the registry that run wrote. Gate 1 rejections cost Gate 1 turns and write `review_turn` ledger rows. Rejected: signing registries, which proves a table was not edited but not that a run produced it. | `test_a_revision_runs_under_gate_1_before_gate_2_sees_it` |
| 09-16 | D25 | **Gate 3's retrieval registry is `lit_review` plus the writer's per-section arXiv search results.** Amends D21. The writer searches arXiv per section and is told to cite from those results (`papersolver.py:349-381`); the archived gated run cites 8 arXiv ids and `lit_review` holds 1 of them, so D21 alone flags 7 real papers. The fabrication targeted is an id nothing retrieved. The archived log holds the ADD_PAPER ids but no search results, so D25's number needs a new run. Decided by Kesh. | `test_the_archived_run_can_only_be_measured_against_its_literature_review`, `test_the_host_retrieval_record_is_read_from_its_own_formats` |
| 09-16 | D26 | **Canonical identifier is the arXiv id, compared version-stripped**; a version mismatch is evidence, not a failure. No DOI fallback: the reference host never sees a DOI, so a cited DOI fails as not retrieved. Decided by Kesh. | `test_versions_are_compared_stripped_and_a_mismatch_is_evidence`, `test_a_cited_doi_fails_as_not_retrieved` |
| 09-16 | D27 | **`style.sections_present` checks sections the host declares at wiring time** (D13 pattern). No default list in `gates/`. Agent Laboratory's adapter declares its writer's list (`papersolver.py:352`). Decided by Kesh. | `test_the_reference_host_declares_its_writers_own_sections` |
| 09-16 | D28 | **Declared limitations are rendered, not authored.** The renderer inserts Gate 2's `declared` block verbatim and Gate 3 checks it is present. Rejected: matching a writer's paraphrase, which no deterministic check can do. Decided by Kesh. | `report.limitations_declared`, see D35 |
| 09-16 | D29 | **Gate 3's loop is `report_loop` in the adapter, not `rig/loop.py`.** Amends D5. `run_loop` is Gate 1-wired (`rig/loop.py:198`); Gate 3 follows Gate 2's as-built shape. `report_loop` takes a registry, not a `ReviewOutcome`, so a host without Gate 2 can use it. `Ledger.loop_summary` counts Gate 2 rows only, since both phases share `divergence.jsonl`. | `test_a_spent_budget_raises_and_emits_nothing`, `test_gate_3_turns_are_not_counted_as_gate_2_reviews` |
| 09-16 | D30 | **Gate 3 is described as mirroring Gate 1**: a flat check list, a reject-and-retry loop back to the agent that wrote the artifact, and a raise on a spent budget. It verifies the manuscript's words and cited sources instead of code. D29 still holds for the code. Replaces "Gate 2's sibling" at `gates/gate3.py:5`. Decided by Kesh. | `gates/gate3.py` module docstring |
| 09-16 | D31 | **Gate 3 gets Gate 1's model layer, rebuilt for manuscripts.** Same structure as Gate 1: an injected `ModelFn` on `Gate3Config`, model findings only through `model_warning` (WARN), REQUIRED FIXES written after the verdict and dropped if ungrounded, spend through `ModelBudget`. The feedback goes to the host's paper writer (`PaperSolver`, `ai_lab_repo.py:294`) the way Gate 1's goes to `MLESolver`. Design approved 09-16 (Q10, Q11, Q13): claim scan WARN only, fixes grounded against the registry's keys, feedback to `PaperSolver`. Decided by Kesh. | `test_a_hostile_model_cannot_move_a_verdict`, `test_the_claim_scan_module_never_names_severity_fail`, `test_a_fix_proposing_an_unrecorded_key_is_dropped_whole` |
| 09-16 | D32 | **Gate 3 reads the retrieved-paper set through `retrieved: Callable[[], set[str]] | None` on `report_loop`**, called after each `write`, because the host fills `section_related_work` during its first write (`papersolver.py:357-367`). `None`: the source checks emit nothing. Rejected: `write` returning papers too, which widens the contract for hosts with no source checks. Decided by Kesh. | `test_the_retrieval_record_is_read_after_each_write` |
| 09-16 | D33 | **Remaining `style.*`**: `style.no_orphan_references` FAIL, only when the manuscript uses `\ref` or `\label`; `style.floats_referenced` WARN; `style.acronyms_defined` dropped, since which acronyms need expanding is a preference D27 refuses without a host-declared list. Decided by Kesh. | `style.no_orphan_references`, `style.floats_referenced` in `gates/gate3.py` |
| 09-16 | D34 | **`style.claim_sections_bound` holds only sections whose heading contains "results"** to at least one `\result{}` token, counted whether or not it resolves. A qualitative discussion is honest writing; a numberless Results section is the evasion. No results heading: the check emits nothing, and `style.sections_present` catches the absence when the host declares it. Rejected: every findings section (fails honest discussions, and an abstract is only seen as `\section{Abstract}`); one token anywhere (lets an empty Results through). Decided by Kesh (Q8). | `test_a_results_section_that_cites_nothing_fails`, `test_only_a_results_section_must_cite_a_measurement` |
| 09-16 | D35 | **How D28 is built.** `declared` is a keyword on `run_gate3`, `gated_report` and `report_loop`, beside `registry`, since both come out of Gate 2's run; not on `Gate3Config`, which is built before Gate 2 runs. The writer places `\limitations{}`; `render_result_tokens(declared=)` replaces it first, inside a LaTeX `verbatim` block, because `_` breaks the build and `%` hides the rest of a line. `report.limitations_declared` FAIL checks every stripped line of the block against the rendered text (the host's when supplied), and emits nothing when `declared` is empty. LaTeX only. Decided by Kesh (Q9). | `test_a_manuscript_without_the_limitations_token_fails`, `test_latex_specials_in_a_limitation_cannot_break_or_hide_it` |
| 09-16 | D36 | **Provider keys stay outside `gates/`.** The key lives only in the function `make_gate_model` returns, one per gate, so spend is counted per gate; Gate 1's child gets a scrubbed environment. No restructure. A test runs all three gates with a sentinel key and fails if any written file or model prompt holds it (it fails with the scrub disabled). Separate function instances share one provider rate limit. Decided by Kesh (Q14). | `test_no_gate_writes_or_sends_the_key` |
| 09-16 | D37 | **G3-M4's hand labels are reviewed before the figure is recorded.** It is the one Gate 3 number with a human label rather than a deterministic check behind it, and the paper quotes it. Decided by Kesh. | **closed 09-16: reviewed and accepted.** `docs/PLAN.md` §5.2 |
| 09-16 | D38 | **Step 8 measures the scanner and does not fix it.** All three readout §6 classes are reported, including the wrong-value ones: `-0.42` scans as `0.42` with the sign dropped, and `1.2e-3` scans as `1.2`. Rejected: fixing negatives and scientific notation first, because the published Gate 1 traceability number came from this scanner reading `.tex` (`gates/prose.py` docstring) and any change restates a measured result. Decided by Kesh. | `rig/gate3_scanner_miss.py`; `gates/prose.py` unchanged |
| 09-16 | D39 | **G3-M4's denominator is the two archived manuscripts, and the §6 probe table is an enumeration of miss classes, not a rate.** No Wilson interval over two documents; the plan's G3-M4 asks for one and claiming it would be the same overclaim as a green check that never ran. Decided by Kesh. | `rig/gate3_scanner_miss.py` prints no interval |
| 09-16 | D40 | **`\begin{abstract}` satisfies a declared "abstract" for `style.sections_present`, and `prose._heading` is not changed.** Presence is a different question from claim scanning. The ungated archived manuscript's 11 literals stay invisible to the scanner and that gap is reported in G3-M4 rather than closed here, because closing it restates a measured result (D38). Decided by Kesh. | `test_a_latex_abstract_environment_counts_as_a_declared_abstract` |
| 09-16 | D41 | **Step 9's HTTP client lives in the adapter, not `rig/`.** Adapters are where host knowledge lives; `rig/` is the model-free scenario loop and never opens a socket. Cache under `.cache/`. A network failure or an uncached miss emits nothing, never a pass. One live arXiv smoke test, marked so the default suite stays offline; arXiv asks for about one request every 3 seconds. Decided by Kesh. | `arxiv_lookup()` in `gates/adapters/agentlab.py`, `tests/test_arxiv_lookup.py` |
| 09-16 | D42 | **Host edits are back in scope, on a branch in `AgentLaboratory-Gemini`, with one gated run budgeted.** Amends the 09-14 scope note. D25's real fabricated-citation number needs a run that logs per-section search results, and Gate 3 has never executed inside the scaffold it claims to be portable to. Decided by Kesh. | **call sites connected** on `feat/gates-d42-connect`: `running_experiments` -> `review_loop(..., first=final)`, `report_writing` -> `report_loop`. Host tests: `tests/test_gates_d42.py`. Live run still budgeted. |
| 09-16 | D43 | **`PLAN.md` comes down to what is built.** `report.bibliography_generated` and `report.citation_metadata_matches` leave §5.1, `report.citations_in_registry` is renamed to the as-built `source.cited_papers_in_registry`, and §5.2's citation row becomes detected-and-measured rather than "eliminated by construction". Rejected: building a registry-emitted bibliography to earn the construction claim, which would change how the host writes citations and cost the portability claim. Decided by Kesh. | `docs/PLAN.md` §5.1, §5.2 |
| 09-16 | D44 | **`CLAUDE.md` §6 is narrowed, not deleted.** Step 6 built the retrieved-id half via `retrieved_arxiv_ids()`; `source.identifiers_resolve` still has no registry. Decided by Kesh. | `CLAUDE.md` §6 |
| 09-16 | D45 | **F11's `SKILL.md` lands before F9.** If E1 installs G.A.T.E.S. through the skill, portability becomes evidence from the experiment instead of an artifact shipped beside it. Doing it after E1 leaves the paper's central claim the only one with no run behind it. Decided by Kesh. | `SKILL.md`, `tests/test_install_skill.py` |
| 09-16 | D50 | **A known fabrication outranks an outage.** `source.identifiers_resolve` asks about every cited id and never returns early on a resolver failure. It degrades to INFO only when nothing was found unresolved; if one id resolves to nothing and another cannot be reached, the check FAILs on the first and reports the second as unchecked. Found by review: the early return let such a manuscript pass, which is duty 1. Consequence for D49: a corrupt cache must not raise either, or it arrives as an outage while the network is fine. | `test_a_fabricated_citation_still_fails_when_another_lookup_breaks`, `test_a_cache_row_missing_its_fields_is_refetched` |
| 09-16 | D49 | **A resolver is asked for the version-stripped id, and a failure raises rather than returning `None`.** Version-stripped per D26: whether v4 specifically exists is what the run read, which `source.cited_papers_in_registry` already judges. The raise-vs-`None` split is what the design rests on: `None` means arXiv has no such paper, so the citation fails; a raise means the question could not be asked, so Gate 3 emits an INFO row saying citations went unchecked and the verdict is untouched. Collapsing them would make an outage reject an honest manuscript. A resolved or absent answer is cached; a failure is not, since a cached outage would keep citations unchecked after the network returned. | `test_a_lookup_that_cannot_reach_the_network_says_so`, `test_a_failed_fetch_is_not_cached` |
| 09-16 | D48 | **G3-M4's headline is the cause, not the rate.** 12 of 15 misses come from one rule: `SKIP_LINE` matches `\ref`, so a findings sentence that points at its own table or figure reports nothing. The readout found this class with a `\cite` probe and called it rare; on real manuscripts `\ref` is the common trigger, because that is how a findings sentence refers to a float. Two further findings: `duplicate_context` is a fourth class the readout did not reach (`context_of` uses `line.find`, so a value stated twice on one line is deduplicated to one), and it understates the literal count without letting a line through. The unreadable `\begin{abstract}` (D40) hides nothing on this corpus, because the ungated run recorded no metrics and its abstract states none; the readout must not imply otherwise. | `rig/gate3_scanner_miss.py`, `tests/test_gate3_m4.py` |
| 09-16 | D47 | **A WARN-only check gets no rig scenario.** The rig proves a reject-fix-accept cycle closes, and a check that never rejects has no cycle. `style.floats_referenced` is covered by unit tests asserting the warning is emitted and the verdict stays PASS. Rejected: a scenario showing a non-blocking warning, which would test the loop's indifference to it rather than the loop. | `rig/gate3_scenarios.py`, nine scenarios |
| 09-16 | D46 | **The `docs/gate3/` blobs stay in this branch's history.** `git add -A` committed them in `8ba2612` before `e34620c` untracked them, so 525 KB of PDF and PNG remain reachable. Purging them needs a force-push, which the working rules forbid, and rewriting published history to reclaim half a megabyte is not worth suspending that rule. Do not rebase them out. Decided by Kesh. | `.gitignore:15` |
| 09-13 | D21 | **Gate 3's retrieval registry is Agent Laboratory's `lit_review`, `ADD_PAPER` entries only.** A paper read with `FULL_TEXT` but never added does not count as retrieved. Identifier is the host's `arxiv_id`. Decided by Kesh. | `retrieved_arxiv_ids` (amended by D25) |

## session log

| Date | SHA | What |
|---|---|---|
| 09-11 | `ad3a416` | `Range.admits` rejects NaN and inf. Nine unbounded-above units were admitting both. |
| 09-11 | `c4d7573` | `auc`/`f1`/`precision`/`recall` bounded; `perplexity >= 1`. |
| 09-11 | `756f7ee` | `coherence.plausibility`, provenance-gated. Renderer + fix directive + completeness guard. |
| 09-11 | `fb59b5b` | `rig/gate2_tier_a_evidence.py`: 12 fixtures, 7/12 before, 12/12 after. |
| 09-11 | `fe40702` | Ceiling 1000 → 500 to avoid collision with `MAX_LEN`. |
| 09-11 | `6c88687` | Canonical `progress.md` adopted; `tests/test_progress.py` makes its counts true. |
| 09-11 | `11da445` | `make_review_context()` closes B7. Gate 2 is reachable from the host. |
| 09-11 | `01591a7` | Tier A scored as a detector: 45 labelled registries, 27/27 and 0/18, Wilson intervals. |
| 09-12 | `369c968` | Tier B: `method_conformance` and `method_traceable`, 13 tests. |
| 09-12 | `b0a2b31` | Tier B scored as two detectors: 29 cases, 12/12 and 0/17, 6/6 and 0/23. |
| 09-12 | `d738e49` | Merged to `main`. 439 tests. |
| 09-13 | `066ee68` | D4: `gate2_semantic.py` deleted, 15 tests removed, combination test rewritten over plan and sources. 424 tests. |
| 09-13 | `65c5ed5` | `make_review_context(plan_fields=...)`: tier B reachable from the host. 425 tests. |
| 09-13 | `85af14c` | Tier C: `rig/gate2_loop.py`, 5 scenarios, 14 tests. Also committed `reports/*.zip` (6.3 MB). 439 tests. |
| 09-14 | `13438ca` | Gate 2 missing list F1-F11, B8 recorded, STATE re-recorded. 439 tests. |
| 09-14 | `59511a0` | B8 closed: `used_by_run` provenance, tier B reason `unused`. 446 tests. |
| 09-14 | `0baf625` | F4 closed: loop body moved to adapter `review_loop`, rig calls it; `declared` set every turn. 447 tests. |
| 09-14 | `12aa1d9` | F6 closed: declared `sources` bound to `lit_review` in `make_review_context` (D23). 450 tests. |
| 09-14 | `60af585` | F10 closed: tier A 27 TP / 18 TN and tier B 12/17, 6/23, 29/29 asserted. 452 tests. |
| 09-14 | `2a3a9a4` | F7 closed: `Ledger.loop_summary()`; `review_loop` rows carry `max_attempts`. 453 tests. |

| 09-16 | `fe4568d` | F12 recorded: Gate 2 to Gate 1 path open. `head` corrected. 453 tests. |
| 09-16 | `77efbac` | F12 closed: `review_loop(context, revise, gate1=)` runs each revision under Gate 1; `revise` returns code; scenarios emit code; scenario 4 is now the B8 decoy. 454 tests. |
| 09-16 | `9bc5923` | Gate 1 runs inside `review_loop` reach the ledger as `review_turn` rows; `loop_summary` unchanged. 455 tests. |
| 09-16 | `02425bd` | Scenario 6 `hand-typed-fix`: a typed number is Gate 1's to reject and costs no Gate 2 turn. M5 now 6 runs, 3/4 resolved. 457 tests (recorded as 456 in `02425bd`, corrected in the next commit). |
| 09-16 | `c2e4551` | STATE total corrected to 457. |
| 09-16 | `68986bc` | Gate 1 budget inside `review_loop`: raises if nothing ran clean, else ends `proceeded` with the last review declared. 459 tests. |
| 09-16 | `5eeffbe` | `review_loop(first=)`: the Gate 1 pass a host already holds is reviewed without re-running; a rejected `first` raises `GateError`. 461 tests. |
| 09-16 | `890f42f` | `ReviewOutcome.registry`: the last reviewed run's registry, the one the writer cites and Gate 3 checks. 463 tests. |
| 09-16 | `444dc4d` | Tier comparison: `rig/gate2_tier_comparison.py`, `tests/test_gate2_tiers.py` (8). A vs A+B: 27/27 both, divergences 0 vs 12/12, unverifiable 0 vs 6/6, 0/35 false rejections both. A+B vs A+B+C over 6 scenarios: fixed 0 vs 3/4, declared 4 vs 1. 471 tests. |
| 09-16 | `12bf4a0` | Docs: README (Gate 2 row, porting step 5, test counts, Gate 2 rig), PLAN.md (as-built note, step 6), CLAUDE.md count 471. 471 tests. |
| 09-16 | `5ab0bf3` | Gate 2 marked complete in this repo; components, D24, next steps (Gate 3 planning). 471 tests. |
| 09-16 | `4508891` | Gate 3 step 1: B4 closed, `ARCHIVED` asserts 29 literals over abstract, results, discussion (red on the old 8 first). D25-D28 recorded, Gate 3 order revised. 471 tests. |
| 09-16 | `343c9fe` | Gate 3 step 2: `make_report_context`, `gated_report`, `report_loop`, `ReportOutcome`; `gate3.RENDERED_FILENAME`; `loop_summary` filtered to Gate 2 (red first: a Gate 3 turn counted as a Gate 2 run). D29. 480 tests. |
| 09-16 | `9b3aaa8` | Gate 3 step 3: `rig/gate3_loop.py`, `rig/gate3_scenarios.py` (`clean`, `typed-literal-fixed`, `unknown-token`, `budget-exhausts`), `tests/test_gate3_loop.py` (11, red first on the spent budget). Registry from a real Gate 2 `clean` run; the provenance test fails on a hand-built registry. README rig line. 491 tests. |
| 09-16 | `8280f7d` | Docs: README and CLAUDE.md counts to 491 (Gate 3: 39). D30-D33 recorded, architecture restated for D30/D31. 491 tests. |
| 09-16 | `ff2382b` | Q12 fix: `_evidence_missing_keys` listed 5 keys and hid the rest; now lists all (red first, 6 keys). Also changes Gate 1 feedback; no frozen log carries the line. 492 tests. |
| 09-16 | `edc5f77` | Gate 3 step 4: `style.claim_sections_bound` (D34), renderer and fix; `prose.sections()`; D14 guard for Gate 3 (fails with a renderer removed); the literal check no longer calls a numberless paper fully cited; docstring per D30/D7; scenario 5 `no-numbers-in-results`. 502 tests. |
| 09-16 | `99a7098` | Gate 3 step 5: `\limitations{}` token and `report.limitations_declared` (D35), renderer and fix; `declared=` through the adapter; rig `Scenario.gate2`, scenario `undeclared-limitation` from Gate 2's `divergence-exhausts`. 513 tests. |
| 09-16 | `9a37134` | Gate 1 fix: with no model, `ModelLayer.ask` recorded a failed call, so a run that printed and failed was told the model "could not be reached" (red first). No call is recorded now; `report.model` is `null`. 31 frozen reports carry the old record, noted in `reports/README.md`. 514 tests. |
| 09-16 | `2963c76` | D31: Gate 3 model layer. `gates/llm_claims.py` (claim scan over findings rows the scanner passed, `\begin{abstract}` included, tokens masked); `Gate3Config.consult_model`; fixes after the verdict with `REPORT_SYSTEM` and `CITABLE KEYS`, dropped whole on an unrecorded `\result{}`; `attach_fixes` moved out of `gate1.py`; `REPORT_GATE_INSTRUCTIONS`; D14 guard covers warnings. 535 tests. |
| 09-16 | `975059b` | Q14 / D36: `tests/test_key_leak.py` plays Gates 1-3 through the adapter with a sentinel key and a fake host `inference`; no file or prompt holds it; verified failing with `_without_credentials` disabled. 536 tests. |
| 09-16 | `7c351cb` | Gate 3 step 6: `source.cited_papers_in_registry` (arXiv ids version-stripped, mismatch as evidence, DOIs fail), `retrieved=` through `run_gate3`/`gated_report`/`report_loop` (read after each write), `retrieved_arxiv_ids()`, fixes grounded against retrieved ids, scenario 4 `fabricated-citation`. Archive: 8 cited, 2 ADD_PAPER, 7 flag under D21; no search results logged. 551 tests. |
| 09-16 | `6881511` | D37-D45 recorded: Kesh accepted all recommendations from the second question round. Next steps restructured, host edits back in scope (D42), F11 before F9 (D45). Docs only. 551 tests. |
| 09-16 | `8ba2612` | Gate 3 step 7a: `style.sections_present` (D27, D40), renderer and fix; `Gate3Config.sections`; `WRITER_SECTIONS` in the adapter, the host's `papersolver.py:352` list minus `scaffold`, and `make_report_context`'s default. `\begin{abstract}` counts, `prose._heading` untouched. Scenario 8 `missing-section`; `Scenario.sections`. Six fixtures opt out with `sections=()` because they are two-section manuscripts testing other checks. 558 tests. |
| 09-16 | `e34620c` | `docs/gate3/` untracked after `git add -A` swept it into `8ba2612`; `.gitignore` rule added so it cannot recur. See D46. 558 tests. |
| 09-16 | `46c9d53` | Session log closed for `8ba2612` and `e34620c`; D46 recorded. 558 tests. |
| 09-16 | `4cf3d32` | Gate 3 step 7b: `style.no_orphan_references` (D33), renderer and fix. A `\ref` with no `\label` fails; an unreferenced label does not, since a float nobody points at is `floats_referenced`'s. `\cref{a,b}` split into targets. Scenario 9 `orphan-reference`. D14 guard fixture trips it; verified biting. 565 tests. |
| 09-16 | `2d76579` | Gate 3 step 7c: `style.floats_referenced` (D33), WARN, renderer only - `_required_fixes` reads failures, so a WARN fix entry would be dead code, as with `report.model_unbound_claims`. Multi-line and starred environments matched; an unlabelled float is not counted. No rig scenario: a WARN check never rejects, so there is no cycle to close (D47). **Step 7 complete.** 571 tests. |
| 09-16 | `9a8ba41` | Gate 3 step 8 (G3-M4): `rig/gate3_m4_labels.py` (hand labels, hash-pinned), `rig/gate3_scanner_miss.py`, `tests/test_gate3_m4.py` (9). **34 of 49 labelled claims detected, 15 missed (30.6%); 40 reported of which 6 are not claims.** 12 of 15 misses are one cause: `SKIP_LINE` matching `\ref` (D48). Scanner not changed (D38). Awaiting Kesh's label review before the figure is published (D37). 580 tests. |
| 09-16 | `426f152` | Gate 3 step 9: `PaperRecord` in `gates/schema.py`; `Gate3Config.lookup`; `source.identifiers_resolve` (FAIL, INFO when the resolver raises), renderer and fix; `arxiv_lookup()` in the adapter (D41) with a `.cache/` disk cache, 3 s spacing, version-stripped ids (D49); `make_report_context(lookup=)`; scenario 10 `unresolvable-citation` with a declarative fake. Live arXiv test passes, skipped unless `GATES_LIVE_ARXIV=1`. `citations_parse` and `metadata_agrees` not built: no input on this host. **Gate 3's check list is complete.** 600 tests. |
| 09-16 | `513b082` | Gate 3 step 10 (docs): `PLAN.md` §5.1/§5.2 brought down to as-built (D43) - the two unbuildable citation checks removed with the reason, §5.2's citation row is detected-and-measured, and the numeric row now says construction is a property of the pipeline with G3-M4 beside it. G3-M4's figure deliberately not recorded there, pending D37. `CLAUDE.md` §6 narrowed (D44). README Gate 3 row and description. Enforcement cells filled for D27, D33, D38-D41, D43, D44. 600 tests. |
| 09-16 | `46ddc85` | Gate 3 step 11: review of `5ab0bf3..HEAD` found two real bugs, both fixed red-test-first. (1) **Duty 1**: `_check_identifiers_resolve` returned on the first resolver exception, discarding identifiers already known unresolved, so a paper citing one fabricated and one unreachable id could PASS. Now every id is asked about and an outage degrades only when nothing was found unresolved (D50). (2) A corrupt cache row raised out of `arxiv_lookup` and reached the gate as an outage; `_as_record` returns `None` and the row is refetched, and an unreadable cache file no longer takes the resolver down. 603 tests. |
| 09-16 | `82c06bb` | **D37 closed: Kesh reviewed and accepted the G3-M4 labels.** Figure published in `PLAN.md` §5.2 (full table, causes, and both scoping notes) and the README Gate 3 paragraph. 603 tests. |
| 09-16 | `c4cbe2d` | **F11 closed: `SKILL.md`**, the install path as a skill (`CLAUDE.md` §3, D45). Five steps: check the scaffold can be gated at all, write the adapter, place three call sites, budget in agent turns, prove it is wired. Carries the D42 connect spec with verified host line numbers (`running_experiments` 349, `make_context` 360, `gated_execute` 388, `report_writing` 279, `PaperSolver` 294, `best_report` 301) and states what this host cannot support. `tests/test_install_skill.py` (21) keeps it true: every entry point, file and number it names is checked, and the documented `write` recipe drives the real `report_loop` through a reject-fix-accept cycle against a fake solver of the host's shape. 624 tests. |
| 09-16 | `dc31079` | Session log closed; branch merged to `main` with `--no-ff`, matching `f8af469` and `d738e49`. Gate 3 is code-complete: nine checks, the loop, the model layer, G3-M4 published, and the install skill. 624 tests. |
| 09-16 | *this* | **D42 call sites connected** on `AgentLaboratory-Gemini` `feat/gates-d42-connect`. `running_experiments` calls `review_loop(..., first=final)`; `report_writing` refuses a missing registry then calls `report_loop` with `arxiv_lookup` and `retrieved_arxiv_ids`. `SKILL.md` worked example brought down to as-built (no drifting line numbers). Live gated run still budgeted. 625 tests. |

Tier A verified per-commit in a throwaway worktree: 395 → 399 → 405 → 415 → 415, each green alone.
