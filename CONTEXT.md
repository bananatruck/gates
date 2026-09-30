# Glossary

The words this project uses with one meaning each.
Decisions behind them are in `progress.md`, D58-D62.

**Level** - the value of `GATES_LEVEL`: 0 all gates off, 1 Gate 1, 2 Gates 1 and 2, 3 all three.
Levels are cumulative, because each gate reads what the one before it produced.
_Avoid_: arm, which is one level of one experiment, and mode.

**Gate 0** - level 0 on a figure: the host exactly as shipped, the control.

**Host** - a research scaffold G.A.T.E.S. installs into: Agent Laboratory today, AI Scientist v2 later.
_Avoid_: system, when the gates are in it.

**System** - a host at a given level, as a benchmark sees it: "Agent Lab" is level 0, "Agent Lab + GATES" is level 3.
GATES alone is never a system; it is a layer.
_Avoid_: "Gate 1+2+3" as a system name.

**Adapter** - the one file per host that knows the host exists: its contexts, model call, level-0 path and retrieval.
The loops every host shares are `gates/pipeline.py`, not the adapter.

**Integrity metric** - the per-benchmark rate of results a system reports that were never measured; lower is better.

**Task score** - the benchmark's own measure of the work, such as pass@1 or MLR-Judge; higher is better.
Every figure pairs the two, so a drop in integrity failures is shown next to what it cost.

**No-input bar** - a bar at a level whose newest gate has nothing to read on that benchmark, such as Gate 3 on CORE-Bench, which has no manuscript.
The gate emits nothing, so the bar repeats the level below it, drawn hatched.

**Dummy row** - a row of `paper/results.csv` holding the expected shape of a run that has not happened.
Any figure that draws one is stamped PLACEHOLDER.

**Claim chain** - a rendered number's provenance, link by link: task, command, log, value, claim.
Gate 1 builds the first four in `CHAIN_LINKS` order, and Gate 3 adds the claim and writes every chain to `claims.json`.
A broken link is counted in `report.claim_chains`, never fatal.

**Mechanism evidence** - what the gates catch and wrongly flag, measured by the rigs and the signed campaign: `paper/mechanism.csv`, figure 8.
It is not a benchmark result, and the figures never mix the two.

**Live tool** - a module in `rig/` that needs a real model or network to measure anything: `tuning`, `corpus` with `--backend`, `posthoc_audit`, `judge`.
The suite runs each against a fake (D61).

**Wave** - one task, one model and one seed at every requested level, started together on one machine by the host's `tools_levels.py`.
Its level 0 and level 3 runs are a pair for the significance test.

**Void wave** - a wave in which a run failed for a reason outside the agent (a stall, a signal, a full disk, an interrupt or an API outage), or in which a condition every sterile run needs did not hold (D67).
It moves whole to `runs-void/`, keeps its logs and metrics for reference, never enters a table, and is rerun at the same seed.
Its cost is spent money and is reported with the study's total.

**Sterile run** - a run that started from nothing but the pinned inputs (task text, configuration, commits, packages), had its own empty caches, working folder and home folder, and held every guarantee the level runner makes, such as its equal share of the GPU (D67).
Only sterile runs enter a table.
_Avoid_: clean run, which says nothing about which conditions held.

**Faked-results candidate** - a paper every eligible judge flags for "Faked Experimental Results"; a person's verdict decides whether it counts (D63).

**Setting** - a value the run was configured with, such as a learning rate or a sweep's lambda: recorded with `record_setting`, cited as `\setting{key}`, never a result (D75).
_Avoid_: calling a setting a result, or a result a setting; the red team's S12 is exactly that.
