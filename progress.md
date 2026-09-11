# Progress

## State

Branch `research/gate2-gate3-readout` off `main` (f8949ea). Draft PR #3 is open and **not
for merge** - review at the Thursday readout.

- c2f09ee - initial `docs/research/gate2-gate3-literature-readout.md`, docs only.
- e211e72 - benchmark comparison, SVG coverage graph, Gate 2 tier table, and 10-minute readout.

Untracked and not mine: `AGENTS.md` at repo root (Codex entry point, verbatim copy of CLAUDE.md
plus one line). Left alone, uncommitted.

## Verified this session

- `.venv/bin/python -m pytest`: **395 passed, 1 skipped** on Python 3.12.4.
- `docs/research/benchmark-fit.svg` parses with `xmllint` and renders at 1000x540.
- AI-Scientist-v2 and Agent Laboratory size snapshots are pinned to exact Git commits; counts use
  tracked production Python files and exclude tests.
- All 23 `gates.*` modules import clean; stdlib-only holds (`inference` in `adapters/agentlab.py:253`
  is the one documented lazy host import).
- Five `SourceClaim` entries constructed from `Sources/` PDFs and run through the real
  `gates.gate2.band_for()`.
- Scanner gaps probed against the real `gates.prose.extract_claims()`, not reasoned about.

## Blockers

- None for the research readout. A Semantic Scholar key remains optional and blocks no deterministic
  implementation work.

## Open findings, ranked (detail in the readout, §-refs there)

1. `Range.admits` returns `True` for NaN and `+inf` on every unbounded-above unit; `OPS["ratio"]`
   returns NaN on a zero denominator. Failing test first, per §5. (readout §5b)
2. `SourceClaim.interval` carries no kind or coverage — a 95% CI and a ±1 SEM both come back
   `origin="reported_interval"`. (§2)
2b. `default_rel_tol=0.05` on a proportion is 11.7x too narrow at 5/12 and **19.1x at 3/15**.
   Route proportions to Wilson. (§3)
3. New Gate 3 check `report.dispersion_supported`: a stated CI/SD/p-value requires n >= 2 in the
   registry. Deterministic, no model. (§2b)
4. `PaperRecord` does not exist; blocks Gate 2 tier B provenance and Gate 3 citation binding.
   Demonstrated working end-to-end in §4 (arXiv fetch + PDF sha256 + all 5 source_ids bound).
   Resolver goes in the adapter, never in `gates/` — a network call breaks determinism the same way
   a model call would. OpenAlex primary, Crossref fallback, arXiv for versions, skip S2. (§4)
5. Scanner: negatives and scientific notation produce *wrong values*, not misses. Integer
   percentages are a bigger hole than the reported `9 points`. LaTeX path cannot change without
   restating the published Gate 1 traceability number. (§6)

## Decisions taken

- **MLR-Bench is the primary end-to-end evaluation.** CORE-Bench covers execution and stochastic
  compatibility, BadScientist is the adversarial no-evidence paper, and SPOT measures the WARN-only
  semantic tier. The graph uses an explicit coverage rubric because the headline percentages are
  not commensurable.
- **G.A.T.E.S. is compared as a validation overlay, not a competing researcher.** It is smaller
  operationally than AI-Scientist-v2 or Agent Laboratory, while its code surface is concentrated in
  evidence, reporting, deterministic verdicts, and feedback-loop integration.
- **Speedup stays unbounded above.** An upper bound is an empirical prior; tier A is for facts about
  the numbers alone. SAGE reports a real 4,700x ratio, so a plausible ceiling false-rejects
  published work. Concern belongs in the relation check, which is exact.
- **MiniCheck punted** from `gates/` (PyTorch vs zero-dependency; and `model_warning()` hardcodes
  WARN so it could not decide anything anyway). Keep as a `rig/` yardstick.
- **The 42% is real and is ScientistOne's, Table 1, about AutoResearchClaw** (5/12 score
  verification; also 3/15 method-code alignment, lowest in the table). An earlier note here said
  it did not exist -- that was from searching AutoResearchClaw's own paper, which does not state
  it. Cite `arXiv:2605.26340`, not `2605.20025`.
- **arXiv API is the PaperRecord identity spine.** Crossref 404s on arXiv DOIs (DataCite holds
  them); Semantic Scholar 429s unauthenticated but is the only resolver that knows a real venue;
  OpenAlex misspelled an author and hides the arXiv id. Only arXiv is version-authoritative.

## Next

Present the 10-minute readout. If implementation is approved afterward, take items 1-3 above first;
they are small, deterministic, and independent of `PaperRecord`. Keep PR #3 unmerged.

## Open decision

Semantic Scholar API key: optional, blocks nothing, deferred until the resolver lands. Everything
in the readout works on keyless APIs today.
