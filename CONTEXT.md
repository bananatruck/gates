# Glossary

The words this project uses with one meaning each.
Decisions behind them are in `progress.md` (D58-D60).

**Level** - the value of `GATES_LEVEL`: 0 all gates off, 1 Gate 1, 2 Gates 1 and 2, 3 all three.
Levels are cumulative, because each gate reads what the one before it produced.
_Avoid_: arm (an arm is one level of one experiment), mode.

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
