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
branch: feature/gate2-feedback-loop
head: 2a3a9a4
head_date: 2026-09-14
tests_total: 471
tests_gate1: 98
tests_gate2: 92
tests_gate3: 19
tests_llm_scan: 21
tests_llm_layer: 14
<!-- STATE:END -->

Named suites above do not cover every test file; `tests_total` is checked against the whole tree,
so a new test file surfaces as a total mismatch even without its own key.

Known environment-dependent result: `test_experiment_child_cannot_read_parent_proc_environment`
**skips** on macOS (`Linux /proc test`) and fails as root on Linux (uid 0 bypasses the
`PR_SET_DUMPABLE` guard via `CAP_SYS_PTRACE`). CI runs non-root Linux so CI is green. See B3.

## architecture

Decided 2026-09-10/11. **Tiers are a Gate 2 mechanism only.** Gate 3 mirrors Gate 1: a flat list of
checks with severities, no tier structure, and Gate 1's feedback loop reused as-is.

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
| Gate 1 — execution validity | complete, evidence frozen | `gated_execute()` | 92 |
| Gate 2 tier A — boundaries | **complete** — ranges, relations, non-finite guard, plausibility ceiling | `run_gate2()` | part of 86 |
| Gate 2 tier B — methodology conformance | **complete** — `method_conformance` FAIL, `method_traceable` WARN; host path via `make_review_context(plan_fields=...)` | `run_gate2()` | part of 86 |
| Gate 2 tier C — loop | **loop closes model-free**, 6 scenarios, exhaustion proceeds declared. Host entry point exists; no host calls it yet | `gates/adapters/agentlab.py` `review_loop()`, driven by `rig/gate2_loop.py` | 15 |
| Gate 3 — `report.*` regex checks | numeric + figure binding done | `run_gate3()` | 19 |
| Gate 3 — `source.*` | **not written** | — | 0 |
| Gate 3 — `style.*` | **not written** | — | 0 |
| Gate 3 loop | **not written**; reuses `rig/loop.py` unchanged | — | 0 |
| LLM scan layer | complete, Gate 1 only | `gates/llm_scan.py` | 21 |
| LLM plumbing (`ModelFn`, budget) | complete, Gate 1 only | `gates/llm.py` | 14 |
| `gates/gate2_semantic.py` | **deleted** 09-13 (D4). Gate 2 is model-free (D19) | - | 15 removed |
| Agent Laboratory adapter | `gated_execute` + `gated_review` via `make_review_context()`, which takes `relations`, `ranges`, `plan_fields`; **`gated_report` absent** | `gates/adapters/agentlab.py` | — |
| Tier A evaluation harness | complete | `rig/gate2_tier_a_eval.py` | 45 labelled registries |

Frozen and checksum-signed — do not edit: `reports/finalized-report-and-results/`.

## gate 2 - missing to complete

Verified 09-14 against `85af14c` and `../AgentLaboratory-Gemini` (`feat/gates-verification-layer`, 7 files uncommitted).
**Closed 09-14:** F4 (`0baf625`); F1 + F3 at adapter level, `review_loop` + `ReviewOutcome.declared` (`0baf625`); F6, `make_review_context(sources=, lit_review=)` (D23, `12aa1d9`); F10, exact published tallies asserted in `tests/test_gate2.py` (`60af585`); F7, `Ledger.loop_summary()` from `review_loop` rows (M5; a run enters the loop only if its first review failed). F12 (09-16), `review_loop(..., gate1=)` runs every revision under Gate 1, so Gate 2 never reviews a registry no run wrote.
**Scope 09-14:** gates repo only. AgentLaboratory-Gemini is a test bed for results and data, not edited. Host-facing items are delivered as adapter entry points a host can call, not as host edits.
Tiers A and B run as one call (`run_gate2`); tier C is `review_loop` in the adapter, which the rig drives. Nothing below exists yet.

| ID | Missing | Evidence | Needs |
|---|---|---|---|
| F2 | Nobody declares `plan_fields` or `relations` in a real run, so tier B never activates and tier A checks ranges only. | D13; the host plan is free text | a declaration source |
| F5 | Setup budgets reach nothing. **Host side only**: both context builders take `max_attempts` and their defaults match `setup.defaults()`. | `gates/setup.py:121` prints JSON only | host passes the chosen budget (out of scope 09-14) |
| F8 | M6 wallclock overhead is not measured. | spec §4 M6 | paired timing, with F9 |
| F9 | E1 not run: MLR-Bench's 10 tasks, gated vs ungated, paired. | spec §5 E1 | F1, F2, model spend |
| F11 | No `SKILL.md` install path, for any gate. | `CLAUDE.md` §3; no `SKILL.md` in the tree | after F1 fixes the call sites |

## blockers

| ID | Blocker | Blocks | Opened |
|---|---|---|---|
| B2 | Gate 3 `source.*` needs network, rate limits, caching, and an offline fallback so the suite stays hermetic. Only non-offline component in the plan. Build order puts it last deliberately. | Gate 3 step 5 | 09-10 |
| B3 | `PR_SET_DUMPABLE` guard is inert under uid 0; not detected, not reported. Skips on macOS. CI runs non-root so CI is green. | portability claim | 09-07 |
| B4 | Gate 3 `ARCHIVED` fixture points at `generated_readme.md` (8 literals) not `generated_report.txt` (29). | Gate 3 headline number | 09-07 |

**Closed:** B8 (decoy hole) - 09-14: Gate 1 records `provenance.used_by_run` from `static_checks.find_unused_record_values`; tier B reports a recorded-but-unread plan value as unverifiable `unused` (WARN, D17). Ceiling: names only, see the `ponytail:` comment. B1 (benchmark selection) — MLR-Bench chosen, see D8. B7 (Gate 2 host entry point) — `make_review_context()` added 09-11; `test_the_host_wiring_path_actually_reaches_gate_2` keeps it reachable. B5 (unpushed WIP loop) - restated 09-13: that loop targets the retired `sources`/`consult_model` API, so tier C is rebuilt from `main`; the old code survives only on the local branch `backup/local-main-f1b9fe2`. B6 (host retrieval registry) - yes: Agent Laboratory keeps `self.lit_review`, entries keyed by `arxiv_id` (`AgentLaboratory/agents.py:574,682`); see D21.

## next steps

1. Gate 3 per `GATE3_implementation_plan.md`, steps 1-9 in order, network work last; `gated_report()` in the adapter; Gate 3 input for `ReviewOutcome.declared` (F3 gate side).
2. `env.parent_proc_guard` INFO check (B3).
3. F9 + F8 with model spend; F11 last. F2 and F5 wait on a host call site (out of scope 09-14).

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
| 09-13 | D21 | **Gate 3's retrieval registry is Agent Laboratory's `lit_review`, `ADD_PAPER` entries only.** A paper read with `FULL_TEXT` but never added does not count as retrieved. Identifier is the host's `arxiv_id`. Decided by Kesh. | pending - Gate 3 `source.cited_papers_in_registry` |

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
| 09-16 | *this* | Tier comparison: `rig/gate2_tier_comparison.py`, `tests/test_gate2_tiers.py` (8). A vs A+B: 27/27 both, divergences 0 vs 12/12, unverifiable 0 vs 6/6, 0/35 false rejections both. A+B vs A+B+C over 6 scenarios: fixed 0 vs 3/4, declared 4 vs 1. 471 tests. |

Tier A verified per-commit in a throwaway worktree: 395 → 399 → 405 → 415 → 415, each green alone.
