# G.A.T.E.S. - working rules

A portable, zero-dependency validity layer for autonomous research scaffolds.
The thesis is that hallucinated results are an information-flow defect, not a model tendency.
Every rule below exists to keep that claim true and provable.

Read `docs/PLAN.md` for the design and `README.md` for what Gate 1 actually measured.
This file is what to do; those two are why.

## 1. What a gate is for

Three duties, in order.
When two conflict, the lower number wins.

1. **No fabricated result passes.**
   A gate would rather stop honest work than let an invented number reach the manuscript.
2. **What is caught goes back through the feedback loop, not to a human.**
   A rejection that does not tell the agent what to fix is a stall, not a gate.
3. **What survives is reported as exactly what was checked, never more.**
   Overclaiming in the report is the same defect the gate exists to catch, committed by us.

## 2. Structural invariants

These are not design preferences.
Changing one invalidates a published claim, so do not change one without saying which claim you are giving up.

- **Python stdlib only.** Zero runtime dependencies in `gates/`, forever.
  No `requests`, no `pydantic`, no provider SDK.
  The model arrives as one injected `ModelFn`, never as an import.
- **Deterministic checks decide the verdict.**
  A model call can never block, fail, or change a verdict.
  This is structural, not conventional: `model_warning()` hardcodes `Severity.WARN` and takes no severity argument.
  Do not add one.
- **A check with no input emits nothing.**
  Never a green check.
  A tier the caller did not configure is absent from the report.
  Absent is honest; a green check that never ran is a lie the paper inherits.
- **A gate never mutates what it checks.**
  It reads the artifact and writes evidence beside it.
  It does not edit the code, the results, or the manuscript it is judging.
- **Two artifacts per gate**, and evidence is append-only.
  The ledger records every attempt, accepted or not.
- **Zero scaffold imports.** `gates/` never imports the host, and never imports `rig/`.
  `pyproject.toml` packages `gates*` only, so the `rig/` half is enforced by packaging rather than by discipline.
  The one host import lives inside a function in `adapters/agentlab.py` and is deliberate: at module level it would break `pip install gates`.
  Adapters are where host knowledge is allowed to live, and even there it is imported lazily.

## 3. Product rules

- **Every gate ships a feedback loop, and both halves are tested.**
  Half one: an actionable report the agent can act on, rendered through `gates/report.py`.
  Half two: a model-free scenario loop in `rig/` proving the reject-fix-accept cycle actually closes without an API key.
  Gate 1 has both. A gate with only one is not finished.
- **Every gate installs into an existing scaffold through one adapter and nothing else.**
  Agent Laboratory is the reference host and the comparison baseline.
  If a check cannot be written without knowing the host, it does not get written.
- **Cost is measured, and measuring it is opt-in.**
  Every gate reports its model spend through `ModelBudget`.
  Users evaluating the gates on their own scaffold can turn on cost measurement to see the gated-vs-ungated delta for themselves.
  It is a feature they enable, not a ceiling that trips mid-run.
  Keep prompts small and calls few because the number is published, not because a limit forbids it.
- **The retry budget is chosen during setup, with the cost stated first.**
  Defaults stay in the code and are tuned for completion and accuracy, not for cost.
  Setup shows the cost warning and takes one integer per gate; an empty field accepts our default.
  The decision is made at wiring time with the tradeoff on screen, rather than sleepwalked past.
  `gates/setup.py` owns the warning text and the prompt; no gate compares its own attempt against the budget.
- **The install path ships as skills.**
  `skills/install-gates` routes any coding agent through adding G.A.T.E.S. to an existing research scaffold, and `skills/gate1-execution`, `gate2-coherence` and `gate3-report` install one gate each.
  `.claude-plugin/` packages them for `/plugin install`, and `tests/test_install_skill.py` holds them to the code.
  Portability is the paper's central claim, so the install process is the claim made executable.

## 4. Per-gate policy on exhaustion

Per gate as designed, and the difference is the point.

| Gate | Budget | On exhaustion |
|---|---|---|
| 1 Execution validity | host-set | raise `GateFailure`. A run that never produced a valid experiment must not produce a paper. |
| 2 Source-result coherence | host-set | proceed, carrying unresolved discrepancies forward as declared limitations. |
| 3 Report validity | host-set | raise `GateFailure`. An unverifiable manuscript is not emitted. |

Budget state lives in the adapter, not in the gate.
No gate compares its own `attempt` against `max_attempts`.

## 5. Build conventions

- Tests run with the repo venv: `.venv/bin/python -m pytest`.
  All of them, every time. The count is `tests_total` in `progress.md`, which the suite checks; the live arXiv test is skipped unless `GATES_LIVE_ARXIV=1`.
- Bug fixes start with a failing test that reproduces the bug.
- A `# limit:` comment marks a known ceiling of a check: the case it does not catch, and what would fix it.
  It sits on the line that has the ceiling, and `progress.md` points at it by file.
  A limit is stated, not hidden; the paper's honesty boundary is built from them.
- `.venv/bin/ruff check .` must pass before a commit; CI runs the same rules from `pyproject.toml`.
- Cache and build output live under `.cache/`. Nothing else belongs in the repo root.
- Do not commit scratch notes, TODO dumps, or generated summary docs unless asked.
- `reports/finalized-report-and-results/` is a frozen, checksum-signed submission package.
  Do not edit anything inside it.
  Its metrics are the vintage they were signed at, and the drift is documented in `reports/README.md`.

## 6. Known exceptions

- **Do not mass-remove em dashes from existing source.**
  New prose uses a plain hyphen, but `GATE_NAME` in `gate1.py`, `gate2.py` and `gate3.py` are published identifiers.
  The Gate 1 string appears in four files inside the signed report package, including captured run logs.
  Rewriting them breaks the match between the code and the frozen evidence.
- **`gates/retrieval.py` is not a paper registry.**
 Despite the name it is BM25 over the log-line exemplar bank in `gates/exemplars.py`.
 The registry of papers the scaffold fetched now exists in half: `retrieved_arxiv_ids()` in the adapter reads the host's `lit_review` and per-section search results, which is enough for `source.cited_papers_in_registry`.
 What does not exist is a corpus of the papers themselves. `PaperRecord` in `gates/schema.py` is its shape, and `source.identifiers_resolve` fills one record at a time through an injected resolver rather than keeping a store.
 Gate 2 tier B literature work still needs the corpus.

## 7. Session workflow and progress tracking

- **Check `progress.md` before starting any session:**
  Always read `progress.md` first to load current project progress, active branch and commit state, completed components, and unresolved blockers into the context window.
- **Update `progress.md` after completing or updating each task:**
  Update `progress.md` immediately upon task completion in a minimal, unsloppy format containing only high-density, necessary context (commit SHAs, verified diffs/tests, architectural decisions, blockers, and exact next steps). No conversational filler or speculative claims.
