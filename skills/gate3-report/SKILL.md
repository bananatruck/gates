---
name: gate3-report
description: Wire Gate 3 (report validity) into a scaffold's writing phase, so every number in the manuscript is rendered from the registry and every citation resolves. Use when installing Gate 3, or when a scaffold's writer types result numbers or cites papers it never retrieved.
---

# Gate 3: report validity

Gate 3 judges a manuscript against the registry of the run it describes.
It opens at `GATES_LEVEL=3`.
Install `gate2-coherence` first: the writer cites the registry Gate 2 reviewed, and states the limitations Gate 2 declared.

**Needs:** a citable registry and its declared limitations: `ReviewOutcome.registry` and `ReviewOutcome.declared`.

**Hands on:** `ReportOutcome.manuscript`, the render Gate 3 admitted. It is the only text the host should publish, and it is `None` unless Gate 3 passed it.

Beside each attempt's report, Gate 3 writes `claims.json`: every rendered claim's chain back to its task, counted in the INFO row `report.claim_chains`.
The task link resolves only if the host passed `task_ref` to Gate 1.

## Declare

| Declaration | Feeds | Why the host must say |
|---|---|---|
| `sections` | `style.sections_present` | which sections a paper needs is the venue's standard |
| `retrieved` | `source.cited_papers_in_registry` | only the host knows what it fetched |
| `lookup` | `source.identifiers_resolve` | `gates/` never opens a socket |

`retrieved` is a **callable**, read after each write, because a scaffold that searches while it writes has not finished retrieving when the first draft appears.
`lookup` may raise; that is how it says the question could not be asked, and Gate 3 turns a raise into an INFO row rather than a rejection.
Returning `None` means the paper does not exist, which is a different answer, and collapsing the two lets a network outage launder a fabricated citation.
`arxiv_lookup(cache_dir=...)` in `gates/adapters/arxiv.py` is a ready `lookup` for arXiv ids.

## Wire it

1. **Prompt.** Add `REPORT_GATE_INSTRUCTIONS` from `gates/pipeline.py` to the writer's notes. It asks for `\result{key}` tokens in place of typed numbers, and `\limitations{}` where the paper discusses its limitations. The renderer substitutes both. Below level 3 the host's writer runs without it, because the tokens are Gate 3's treatment.
2. **Context.** Build it with your adapter's `make_report_context(...)`.
3. **Call site.** When `gate_level() >= 3`: `written = report_loop(ctx, write, registry=outcome.registry, declared=outcome.declared, retrieved=...)`.
4. **`write(feedback)`** returns the next manuscript with its tokens intact, or `None` to stop.
   - The first call returns the host's first draft at its usual effort, so a gated first draft gets no fewer improvement steps than the ungated writer.
   - Later calls void the rejected draft's reward score, append the feedback to the notes as text, run one improvement step, and return the new draft, or `None` if nothing new was written.
5. **Publish** `written.manuscript` and nothing else.
6. **Spent budget.** The loop **raises** `GateFailure`. An unverifiable manuscript is not emitted.

Done when, at `GATES_LEVEL=3`, a draft with a typed result number goes back to the writer, and the manuscript the host saves is the rendered one Gate 3 admitted.

## Prove it

- `test_the_host_wiring_path_actually_reaches_gate_3` in `tests/test_gate3.py` is the reachability pattern.
- `rig/gate3_loop.py` drives ten scenarios and needs no network and no model.
