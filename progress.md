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
branch: research/gate2-gate3-readout
head: fe40702
head_date: 2026-09-11
tests_total: 424
tests_gate1: 92
tests_gate2: 85
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
| B | **methodology conformance** — declared plan vs recorded run | `plan_fields` |
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
| Gate 2 tier A — boundaries | **complete** — ranges, relations, non-finite guard, plausibility ceiling | `run_gate2()` | part of 85 |
| Gate 2 tier B — methodology conformance | **to build**; blocked on B7 | `run_gate2()` | 0 |
| Gate 2 tier C — loop | **not written**; mirrors `rig/loop.py` | — | 0 |
| Gate 3 — `report.*` regex checks | numeric + figure binding done | `run_gate3()` | 19 |
| Gate 3 — `source.*` | **not written** | — | 0 |
| Gate 3 — `style.*` | **not written** | — | 0 |
| Gate 3 loop | **not written**; reuses `rig/loop.py` unchanged | — | 0 |
| LLM scan layer | complete, Gate 1 only | `gates/llm_scan.py` | 21 |
| LLM plumbing (`ModelFn`, budget) | complete, Gate 1 only | `gates/llm.py` | 14 |
| `gates/gate2_semantic.py` | **scheduled for deletion — D4** | — | ~15 of 85 |
| Agent Laboratory adapter | `gated_execute` works; **`gated_review` unreachable (B7)**; `gated_report` absent | `gates/adapters/agentlab.py` | — |
| Tier A evidence harness | complete | `rig/gate2_tier_a_evidence.py` | 12 fixtures |

Frozen and checksum-signed — do not edit: `reports/finalized-report-and-results/`.

### Deleting `gate2_semantic.py` — what it touches

Call sites to remove: `gates/gate2.py:53` (import), the tier C block in `run_gate2`, `Gate2Config`
fields `consult_model` and `claims`, and `tests/test_gate2.py:15` plus the source-reading test.

**Two tests must be rewritten, not dropped:**

- `test_each_tier_combination_emits_exactly_its_own_checks` — the absent-never-green guarantee
  across tier combinations. Rewrite as an A/B version. It now also carries `coherence.plausibility`
  in all four expected lists.
- `test_the_model_is_never_shown_a_value_the_registry_lacks` — a real safety property. If any model
  path survives in Gate 2, this survives with it; if none does, record in the decision log that
  Gate 2 is model-free by construction.

Re-record the `STATE` block after deleting, or `tests/test_progress.py` fails.

## blockers

| ID | Blocker | Blocks | Opened |
|---|---|---|---|
| B7 | **Gate 2 has no working host entry point.** `make_context()` builds a `Gate1Config`; `gated_review()` hands it to `run_gate2()`; that raises `AttributeError: 'Gate1Config' object has no attribute 'ranges'`. Only hand-built configs work, which is what the tests do. Needs `make_review_context()`. | Gate 2 tier B | 09-11 |
| B2 | Gate 3 `source.*` needs network, rate limits, caching, and an offline fallback so the suite stays hermetic. Only non-offline component in the plan. Build order puts it last deliberately. | Gate 3 step 5 | 09-10 |
| B3 | `PR_SET_DUMPABLE` guard is inert under uid 0; not detected, not reported. Skips on macOS. CI runs non-root so CI is green. | portability claim | 09-07 |
| B4 | Gate 3 `ARCHIVED` fixture points at `generated_readme.md` (8 literals) not `generated_report.txt` (29). | Gate 3 headline number | 09-07 |
| B5 | WIP loop code unpushed, reason recorded as "code limitations" — needs restating before it can be planned around. | tier C build | 09-10 |
| B6 | Does the host expose its retrieval registry in a readable form? If not, `source.cited_papers_in_registry` — the highest-value offline citation check — has no input. | Gate 3 step 4 | 09-11 |

**Closed:** B1 (benchmark selection) — MLR-Bench chosen, see D8.

## next steps

1. `make_review_context()` in `gates/adapters/agentlab.py`, closing B7. Prerequisite for tier B.
2. Build Gate 2 tier B per `GATE2_implementation_spec.md` §2, with D10–D13 applied.
3. Delete `gate2_semantic.py` per D4; rewrite the two tests named above; re-record `STATE`.
4. Land `rig/gate2_loop.py` + scenarios + tests from the working tree, or restate B5.
5. Build Gate 3 per `GATE3_implementation_plan.md`, steps 1-9 in order. Network work last.
6. Add `gated_report()` to `gates/adapters/agentlab.py`.
7. Emit an `env.parent_proc_guard` INFO check recording whether the guard actually held (B3).

## decision log

Append-only. One line each: date, decision, where it is enforced.

| Date | ID | Decision | Enforced by |
|---|---|---|---|
| — | D1 | Deterministic checks alone decide verdicts | `gates/gate2.py`, `gates/gate3.py` |
| — | D2 | A check with no input emits nothing — absent, never green | tier derivation from supplied args, not flags |
| — | D3 | `gates/` is stdlib-only at runtime | `.github/workflows/tests.yml` stdlib check |
| 09-10 | D4 | **`gate2_semantic.py` is deleted.** Tier C is the loop; semantic checks have no tier. | pending — see deletion notes above |
| 09-10 | D5 | Gate 3 reuses Gate 1's structure and loop; only check contents differ | `rig/loop.py` reuse |
| 09-10 | D6 | Claim entailment (MiniCheck) deferred past this paper | scope |
| 09-10 | D7 | Tiers are Gate 2 only. Gate 3 has a flat check list, like Gate 1. | `gates/gate3.py` |
| 09-11 | D8 | **Benchmark is MLR-Bench** (arXiv 2505.19955). CORE-Bench rejected: only 17 of its 181 task questions have stochastic answers. | `GATE2_implementation_spec.md` |
| 09-11 | D9 | **Tier B is methodology conformance**, not literature tolerance. `SourceClaim`/`Band`/`band_for` stay in the tree, unwired. | `GATE2_implementation_spec.md` §2 |
| 09-11 | D10 | **The speedup ceiling is gated on provenance, not magnitude.** A speedup a declared `Relation` derives is exempt at any size; SAGE reports a real 4,700x. Separate check `coherence.plausibility`, so `range_valid` stays provable. | `gates/gate2.py` `_check_plausibility` |
| 09-11 | D11 | **`IMPLAUSIBLE_SPEEDUP = 500.0`**, not 1000, to keep it clear of `MAX_LEN = 1000`, the stdout truncation this project diagnosed. Declared, not derived; reaches the report as `ceiling_origin`. | `gates/gate2.py` |
| 09-11 | D12 | **Bounded scores are units, never metric names.** `auc`/`f1`/`precision`/`recall`/`perplexity` in `UNIT_RANGES`. Name-based inference stays refused. | `test_a_metric_name_alone_never_implies_a_range` |
| 09-11 | D13 | **Tier B declarations arrive at wiring time**, not by parsing plan prose. The host passes `plan_fields`; `gates/` never reads a plan. | pending — blocked on B7 |
| 09-11 | D14 | **Every check id must have an evidence renderer and a fix directive.** A keyed lookup that misses returns nothing and the agent gets a rejection it cannot act on. | `test_every_check_gate2_emits_can_be_rendered_and_has_a_fix` |

## session log

| Date | SHA | What |
|---|---|---|
| 09-11 | `ad3a416` | `Range.admits` rejects NaN and inf. Nine unbounded-above units were admitting both. |
| 09-11 | `c4d7573` | `auc`/`f1`/`precision`/`recall` bounded; `perplexity >= 1`. |
| 09-11 | `756f7ee` | `coherence.plausibility`, provenance-gated. Renderer + fix directive + completeness guard. |
| 09-11 | `fb59b5b` | `rig/gate2_tier_a_evidence.py`: 12 fixtures, 7/12 before, 12/12 after. |
| 09-11 | `fe40702` | Ceiling 1000 → 500 to avoid collision with `MAX_LEN`. |

Tier A verified per-commit in a throwaway worktree: 395 → 399 → 405 → 415 → 415, each green alone.
