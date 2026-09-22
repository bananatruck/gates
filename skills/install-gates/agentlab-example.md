# Worked example: Agent Laboratory

The reference host, and the comparison baseline.
It is already wired (D42).
The adapter is `gates/adapters/agentlab.py`; the call sites are in the host's `ai_lab_repo.py`, as built:

| Phase | Method | What it calls |
|---|---|---|
| running experiments | `running_experiments` | `make_context`, then `gated_execute` on the winning code, then `review_loop(..., first=final)` if that run passed and `gate_level() >= 2`. `reviser_from_mle_solver` is the `revise` callback. It records `gate1_passed`, because at level 1 no Gate 2 review exists to say the run was verified. |
| report writing | `report_writing` | Refuses if an open Gate 1 left no verified run. At level 3, `report_loop` with `arxiv_lookup`, `writer_from_paper_solver`, and `retrieved_arxiv_ids(self.phd.lit_review, solver.section_related_work)`. Below level 3, the host's own writer, on the verified evidence. |

The host's callbacks as first built (`1966e17`) broke all three callback rules in the gate skills: they returned the reward-best entry, kept notes as a list, and handed the writer the pre-review code.
Fixed in `1245dea` (F13-F15 in `progress.md`).
The fakes that tested them adopted whatever the next solver step produced, which is why the suite never saw it: a fake that does not rank the way the host ranks cannot fail this way.

## The writing callback

`writer_from_paper_solver` wraps the host's solver.
The first call runs `solver.initial_solve()`, then the host's usual number of `solver.solve()` steps, and returns `"\n".join(solver.best_report[0][0])`.
Each later call does three things:

1. **Voids the rejected draft's reward score**, setting it to `float("-inf")`. The solver keeps one best draft and replaces it only on a higher score, so without this a fix the reward model likes less is discarded and the rejected draft is resubmitted until the budget raises.
2. **Appends the feedback to `solver.notes` as text**, `f"{solver.notes}\n{feedback}"`. The host interpolates notes straight into its prompt, and a list renders as its repr: every `\result{}` gains a second backslash and the rejection collapses onto one line.
3. **Runs one `solver.solve()` and returns the draft it produced**, or `None` if the draft still scores `-inf`, since nothing new was written.

`tests/test_install_skill.py` drives this recipe through the real `report_loop` against a solver that ranks the way the host does.
`reviser_from_mle_solver` follows the same three rules against `solver.best_codes`, tracking the entry it last returned, since the experiment solver keeps two.
Retrieval reads the host's two formats: `lit_review` entries keyed `arxiv_id`, and per-section arXiv search results as text carrying `arXiv paper ID:` lines.

## What this host cannot support, and why that is fine to report

- **No bibliography.** It cites inline as `(arXiv 2308.11483v1)`, so `source.citations_parse` and `source.metadata_agrees` have no input and emit nothing. Do not synthesise one to make a check run.
- **Free-text plans.** `extract_plan_fields` has a judge model read them when the host runs with `--judge-backend`, a different model from the one under test. Each field is model-authored, so a divergence warns and cannot fail the run; without a judge, tier B stays silent.
- **No DOIs.** The canonical identifier is the arXiv id, compared version-stripped.
