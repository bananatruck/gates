# Progress

## State

Branch `docs/gate2-gate3-literature-readout` off `main` (f8949ea). PR #2, open, **not for merge** —
review at the Thursday readout.

- c2f09ee — `docs/research/gate2-gate3-literature-readout.md`, 556 lines, docs only.

Untracked and not mine: `AGENTS.md` at repo root (Codex entry point, verbatim copy of CLAUDE.md
plus one line). Left alone, uncommitted.

## Verified this session

- All 23 `gates.*` modules import clean; stdlib-only holds (`inference` in `adapters/agentlab.py:253`
  is the one documented lazy host import).
- Five `SourceClaim` entries constructed from `Sources/` PDFs and run through the real
  `gates.gate2.band_for()`.
- Scanner gaps probed against the real `gates.prose.extract_claims()`, not reasoned about.

## Blockers

- **Suite not run.** No `.venv` in the repo and no `pytest` on any interpreter on this machine.
  CLAUDE.md §5 documents `.venv/bin/python -m pytest` (396 tests). The venv needs recreating before
  any code change lands.

## Open findings, ranked (detail in the readout, §-refs there)

1. `Range.admits` returns `True` for NaN and `+inf` on every unbounded-above unit; `OPS["ratio"]`
   returns NaN on a zero denominator. Failing test first, per §5. (readout §5b)
2. `SourceClaim.interval` carries no kind or coverage — a 95% CI and a ±1 SEM both come back
   `origin="reported_interval"`. All five entries above are typed wrong today. (§2)
3. New Gate 3 check `report.dispersion_supported`: a stated CI/SD/p-value requires n >= 2 in the
   registry. Deterministic, no model. (§2b)
4. `PaperRecord` does not exist; blocks Gate 2 tier B provenance and Gate 3 citation binding.
   Resolver goes in the adapter, never in `gates/` — a network call breaks determinism the same way
   a model call would. OpenAlex primary, Crossref fallback, arXiv for versions, skip S2. (§4)
5. Scanner: negatives and scientific notation produce *wrong values*, not misses. Integer
   percentages are a bigger hole than the reported `9 points`. LaTeX path cannot change without
   restating the published Gate 1 traceability number. (§6)

## Decisions taken

- **Speedup stays unbounded above.** An upper bound is an empirical prior; tier A is for facts about
  the numbers alone. SAGE reports a real 4,700x ratio, so a plausible ceiling false-rejects
  published work. Concern belongs in the relation check, which is exact.
- **MiniCheck punted** from `gates/` (PyTorch vs zero-dependency; and `model_warning()` hardcodes
  WARN so it could not decide anything anyway). Keep as a `rig/` yardstick.
- **AutoResearchClaw's "42% on audit" does not exist.** `42.9%` is the Thorough HITL acceptance
  rate. Do not cite it. The 42% belongs to SAGE.

## Next

Items 1-3 above are small, deterministic, and independent of the `PaperRecord` work. Item 1 first.
Recreate the venv before touching code.
