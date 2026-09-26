"""Every figure in the paper, drawn from results.csv and mechanism.csv and nothing else.

    python3 paper/figures.py        # writes paper/figures/*.png

Figures 1-7 are the benchmark evaluation, from results.csv. Figure 8 is the
mechanism evidence already measured, from mechanism.csv, where every row names
the rig or signed report that produced it.

Filling in a real result means replacing its row in results.csv (status
``dummy`` becomes ``measured``) and running this again. A figure that still
draws a dummy row is stamped PLACEHOLDER, so an expected shape cannot be
mistaken for a result.

Needs matplotlib. This is not part of the gates package, whose stdlib-only rule
covers ``gates/`` alone.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"

# The dataviz reference palette. The level ramp passes its ordinal checks, and
# gray-vs-blue passes CVD, normal-vision and 3:1 contrast (validate_palette.js).
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
LEVEL_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab"]  # more gates, darker
ALONE, GATED = MUTED, "#2a78d6"

LEVELS = ["L0", "L1", "L2", "L3"]
LEVEL_NAMES = ["Gate 0\n(all off)", "Gate 1", "Gate 1+2", "Gate 1+2+3"]
SYSTEMS = ["AI Scientist v2", "Agent Lab", "ScientistOne"]
METRICS = {
    "CORE-Bench": {
        "integrity": "Answers not traceable to a recorded value (%)  lower is better",
        "task": "Accuracy, pass@1 on Hard (%)  higher is better",
    },
    "MLR-Bench": {
        "integrity": "Papers with faked experimental results (%)  lower is better",
        "task": "MLR-Judge overall score (1-10)  higher is better",
    },
    "BadScientist": {
        "integrity": "Manipulated manuscripts emitted and accepted (%)  lower is better",
        "task": "Honest manuscripts admitted (%)  higher is better",
    },
    "Audit": {
        "integrity": "Released papers citing a paper that does not resolve (%)  lower is better",
    },
}

with open(HERE / "results.csv", newline="") as _f:
    ROWS = list(csv.DictReader(_f))
drawn: set[int] = set()  # every figure, for the completeness check
shown: set[int] = set()  # the figure being drawn, for its footer


def find(benchmark, system, arm, metric):
    for i, row in enumerate(ROWS):
        if (row["benchmark"], row["system"], row["arm"], row["metric"]) == (
            benchmark, system, arm, metric,
        ):
            drawn.add(i)
            shown.add(i)
            return row
    return None


def refs(benchmark, metric):
    return [find(benchmark, r["system"], "ref", metric) for r in ROWS
            if r["benchmark"] == benchmark and r["arm"] == "ref" and r["metric"] == metric]


def is_percent(benchmark, metric):
    return not (benchmark == "MLR-Bench" and metric == "task")


def style(ax, benchmark, metric):
    ax.set_facecolor(SURFACE)
    top = 100 if is_percent(benchmark, metric) else 10
    ax.set_ylim(0, top * 1.18)  # headroom for the label over a full bar
    ax.set_yticks([top * i / 5 for i in range(6)])
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.set_title(METRICS[benchmark][metric], loc="left", fontsize=9, color=INK)


def bar(ax, x, row, color, width=0.62):
    """One bar with its interval and label. Returns the row's status."""
    if row is None or row["status"] == "not_reported":
        ax.text(x, 2, "not\nreported", ha="center", va="bottom", fontsize=7, color=MUTED)
        return "not_reported"
    value = float(row["value"])
    no_input = row["no_input"] == "1"
    ax.bar(x, value, width, color=SURFACE if no_input else color,
           edgecolor=MUTED if no_input else SURFACE, hatch="///" if no_input else None,
           linewidth=1 if no_input else 2)
    top = value
    if row["ci_low"]:
        low, high = float(row["ci_low"]), float(row["ci_high"])
        ax.vlines(x, low, high, color=INK2, linewidth=1)
        top = high
    unit = "%" if is_percent(row["benchmark"], row["metric"]) else ""
    tag = {"published": "\npublished", "measured": ""}.get(row["status"], "")
    if no_input:
        tag = "\nno input"
    ax.text(x, top + ax.get_ylim()[1] * 0.02, f"{value:g}{unit}{tag}",
            ha="center", va="bottom", fontsize=7.5, color=INK)
    return row["status"]


def reference_lines(ax, benchmark, metric):
    handles = []
    for row in refs(benchmark, metric):
        value = float(row["value"])
        ax.axhline(value, color=INK2, linewidth=0.8)
        handles.append(Line2D([], [], color=INK2, linewidth=0.8,
                              label=f"{row['system']}, published: {value:g}%"))
    if handles:
        ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=7, labelcolor=INK2)


def finish(fig, statuses, name, note=""):
    sources = sorted({r["source"] for i, r in enumerate(ROWS) if i in shown and r["status"] == "published"})
    shown.clear()
    footer = "  |  ".join(filter(None, [note, *sources]))
    fig.text(0.01, 0.01, footer, fontsize=6.5, color=MUTED, ha="left", va="bottom")
    if "dummy" in statuses:
        fig.text(0.5, 0.5, "PLACEHOLDER", fontsize=64, color=INK, alpha=0.07,
                 rotation=18, ha="center", va="center", weight="bold")
    fig.savefig(OUT / name, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def level_figure(benchmark, name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    statuses = []
    for ax, metric in zip(axes, ("integrity", "task"), strict=True):
        style(ax, benchmark, metric)
        for i, level in enumerate(LEVELS):
            statuses.append(bar(ax, i, find(benchmark, "Agent Lab", level, metric), LEVEL_RAMP[i]))
        ax.set_xticks(range(4), LEVEL_NAMES)
        ax.set_xlim(-0.6, 3.6)
        reference_lines(ax, benchmark, metric)
    fig.suptitle(f"{benchmark}: Agent Lab at each GATES_LEVEL", x=0.01, ha="left",
                 fontsize=11, weight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    finish(fig, statuses, name, "Hatched: the newest gate has no input on this benchmark, so it emits nothing.")


def compare_figure(benchmark, name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    statuses = []
    for ax, metric in zip(axes, ("integrity", "task"), strict=True):
        style(ax, benchmark, metric)
        for g, system in enumerate(SYSTEMS):
            gated = find(benchmark, system, "L3", metric)
            if gated is None:  # nothing to pair with, so the one cell sits centred
                statuses.append(bar(ax, g, find(benchmark, system, "L0", metric), ALONE, 0.36))
                continue
            statuses.append(bar(ax, g - 0.19, find(benchmark, system, "L0", metric), ALONE, 0.36))
            statuses.append(bar(ax, g + 0.19, gated, GATED, 0.36))
        ax.set_xticks(range(len(SYSTEMS)), SYSTEMS)
        ax.set_xlim(-0.6, len(SYSTEMS) - 0.4)
        reference_lines(ax, benchmark, metric)
    fig.legend(handles=[Patch(color=ALONE, label="system alone (GATES_LEVEL=0)"),
                        Patch(color=GATED, label="system + GATES (GATES_LEVEL=3)")],
               loc="upper right", frameon=False, fontsize=8, ncol=2, labelcolor=INK2)
    fig.suptitle(f"{benchmark}: each system alone and with GATES", x=0.01, ha="left",
                 fontsize=11, weight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    finish(fig, statuses, name)


def audit_figure(name):
    fig, ax = plt.subplots(figsize=(7.5, 3.8), facecolor=SURFACE)
    style(ax, "Audit", "integrity")
    statuses = [bar(ax, g, find("Audit", s, "audit", "integrity"), ALONE) for g, s in enumerate(SYSTEMS)]
    for row in refs("Audit", "integrity"):
        g, value = SYSTEMS.index(row["system"]), float(row["value"])
        ax.hlines(value, g - 0.4, g + 0.4, color=INK, linewidth=1.5)
        ax.text(g + 0.42, value, f"MLR-Bench, published: {value:g}%", va="center", fontsize=7, color=INK2)
    ax.set_xticks(range(len(SYSTEMS)), SYSTEMS)
    fig.suptitle("Released papers, audited after the fact by Gate 3",
                 x=0.01, ha="left", fontsize=10, weight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    finish(fig, statuses, name)


def mechanism_figure(name):
    """Measured already: what each gate catches, and what it wrongly flags.

    One horizontal bar per measure, each with its denominator and, where the
    source reports one, its Wilson interval. No row here is a dummy.
    """
    import sys

    sys.path.insert(0, str(HERE.parent))
    from rig.stats import wilson

    with open(HERE / "mechanism.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    labels = []
    for y, row in enumerate(rows):
        k, n = int(row["k"]), int(row["n"])
        value = 100 * k / n
        off = row["arm"].endswith("off")
        ax.barh(y, value, 0.62, color=ALONE if off else GATED)
        end = value
        if row["interval"] == "yes":
            lo, hi = wilson(k, n)
            ax.hlines(y, 100 * lo, 100 * hi, color=INK2, linewidth=1)
            end = 100 * hi
        ax.text(end + 1.5, y, f"{k}/{n}", va="center", fontsize=7.5, color=INK)
        arm = f", {row['arm']}" if row["arm"] else ""
        labels.append(f"{row['panel']}: {row['measure']}{arm}")
    ax.set_yticks(range(len(rows)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 115)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=7.5, length=0)
    ax.set_xlabel("%  (whisker: Wilson 95% interval)", fontsize=8, color=INK2)
    fig.suptitle("Mechanism evidence, measured: what each gate catches and what it wrongly flags",
                 x=0.01, ha="left", fontsize=10, weight="bold", color=INK)
    ax.legend(handles=[Patch(color=GATED, label="gate on"), Patch(color=ALONE, label="Gate 1 off, the host as shipped")],
              loc="lower right", frameon=False, fontsize=7, labelcolor=INK2)
    fig.text(0.01, 0.01, "Sources: paper/mechanism.csv. Gate 1 from the signed 08-15 campaign; "
             "Gates 2 and 3 from the model-free rigs.", fontsize=6.5, color=MUTED)
    fig.subplots_adjust(left=0.36, right=0.98, top=0.9, bottom=0.12)
    fig.savefig(OUT / name, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return len(rows)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for number, benchmark in enumerate(("CORE-Bench", "MLR-Bench", "BadScientist"), start=1):
        slug = benchmark.lower().replace("-", "")
        level_figure(benchmark, f"fig{number}_{slug}_levels.png")
        compare_figure(benchmark, f"fig{number + 3}_{slug}_compare.png")
    audit_figure("fig7_audit.png")
    mechanism_figure("fig8_mechanism.png")
    # A row no figure reads is a result that silently never reaches the paper.
    missing = [f"{r['benchmark']}/{r['system']}/{r['arm']}/{r['metric']}"
               for i, r in enumerate(ROWS) if i not in drawn]
    assert not missing, f"rows no figure draws: {missing}"
    print(f"{len(ROWS)} rows drawn into {len(list(OUT.glob('*.png')))} figures in {OUT}")
