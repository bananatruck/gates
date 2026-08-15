"""Charts for the Gate 1 report and deck.

Every number here comes from a measurement recorded in this repository or from a
cited paper in the source set. Nothing is illustrative.

Palette is the validated categorical set (light mode, surface #fcfcfb):
blue #2a78d6, orange #eb6834, aqua #1baf7a. Adjacent-pair CVD separation was
checked with the validator rather than by eye — worst pair deutan dE 9.2,
normal-vision dE 27.6. Aqua sits below 3:1 contrast on this surface, so every
bar that uses it carries a visible value label.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
SURFACE = "#fcfcfb"
GRID = "#e6e5e0"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _finish(ax, title, subtitle=None):
    """Title above subtitle, both above the axes, neither overlapping.

    set_title plus a text() at a nearby axes fraction collided; both are drawn
    explicitly at separated heights instead.
    """
    ax.text(0, 1.16 if subtitle else 1.04, title, transform=ax.transAxes,
            color=INK, fontsize=12, fontweight="bold", va="bottom")
    if subtitle:
        ax.text(0, 1.045, subtitle, transform=ax.transAxes,
                color=MUTED, fontsize=8.5, va="bottom")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    path = ASSETS / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")
    return path


# --------------------------------------------------------------------------- #


def chart_check_inventory():
    """What Gate 1 checks, by family and severity. Counted from gate1.py."""
    families = ["static", "exec", "env", "results", "logs", "output/report"]
    fail = [3, 3, 2, 4, 0, 0]
    warn = [0, 1, 1, 2, 1, 0]
    info = [0, 0, 1, 0, 0, 2]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bottom = [0] * len(families)
    for values, colour, label in (
        (fail, BLUE, "FAIL — blocks the run"),
        (warn, ORANGE, "WARN — reported, never blocks"),
        (info, AQUA, "INFO — provenance only"),
    ):
        ax.bar(families, values, 0.6, bottom=bottom, color=colour, label=label,
               edgecolor=SURFACE, linewidth=2)
        for i, v in enumerate(values):
            if v:
                ax.text(i, bottom[i] + v / 2, str(v), ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
        bottom = [b + v for b, v in zip(bottom, values)]

    _finish(ax, "Gate 1 check inventory — 20 checks in 6 families",
            "Severity decides the verdict: FAIL blocks, WARN and INFO never can")
    ax.set_ylabel("checks")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=3)
    return save(fig, "checks.png")


def chart_scanner(bench: dict | None):
    """Log-scanner precision and recall: deterministic vs deterministic+model."""
    labels, prec, rec = ["deterministic\n(regex only)"], [1.0], [0.529]
    if bench:
        for key, name in (("qwen_fewshot_0", "+ qwen3:8b\nno few-shot"),
                          ("qwen_fewshot_3", "+ qwen3:8b\nfew-shot k=3")):
            if key in bench:
                labels.append(name)
                prec.append(bench[key]["precision"])
                rec.append(bench[key]["recall"])

    x = range(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    b1 = ax.bar([i - width / 2 for i in x], prec, width, color=BLUE,
                label="precision", edgecolor=SURFACE, linewidth=2)
    b2 = ax.bar([i + width / 2 for i in x], rec, width, color=ORANGE,
                label="recall", edgecolor=SURFACE, linewidth=2)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontsize=8.5, color=INK)

    ax.axhline(1.0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(len(labels) - 0.45, 1.012, "precision floor", fontsize=7.5, color=MUTED,
            ha="right")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.16)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    _finish(ax, "Log-scanner accuracy on the 68-line labelled corpus",
            "Precision is the floor: a false positive puts a non-issue into the paper")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=2)
    return save(fig, "scanner.png")


def chart_compression():
    """What the digest removes before the model reads a log."""
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    stages = ["raw capture", "after shape collapse"]
    chars = [14096, 578]
    bars = ax.barh(stages, chars, 0.5, color=[MUTED, BLUE],
                   edgecolor=SURFACE, linewidth=2)
    for bar, v in zip(bars, chars):
        ax.text(v + 250, bar.get_y() + bar.get_height() / 2, f"{v:,} chars",
                va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 16800)
    ax.invert_yaxis()
    _finish(ax, "Prompt sent to the log scanner — 95.9% smaller",
            "203 non-blank lines collapse to 4 distinct shapes; lossless in distinct content")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("characters")
    return save(fig, "compression.png")


def chart_ablation(run: dict | None = None):
    """The live ablation: gate on vs gate off, same task and models."""
    if run:
        g, u = run["arms"]["gated"], run["arms"]["ungated"]
        turns = [g["turns"], u["turns"]]
        gv, uv = g["verification"], u["verification"]
        notes = [
            f"accepted at turn {g['accepted_at']}\n{len(gv['matched'])} of "
            f"{len(gv['reported'])} values reproduced" if g["accepted"]
            else "never accepted",
            f"accepted at turn {u['accepted_at']}\n{len(uv['reported'])} values "
            f"recorded" if u["accepted"] else "never accepted",
        ]
    else:
        turns, notes = [2, 3], ["converged", "never converged"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    arms = ["Gate 1 on", "Gate 1 off\n(upstream rule)"]
    colours = [BLUE, ORANGE]
    bars = ax.bar(arms, turns, 0.5, color=colours, edgecolor=SURFACE, linewidth=2)
    for bar, v, note in zip(bars, turns, notes):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.08, f"{v} turns",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
        ax.text(bar.get_x() + bar.get_width() / 2, v / 2, note, ha="center",
                va="center", fontsize=8.5, color="white")
    ax.set_ylim(0, max(turns) + 2)
    ax.set_ylabel("engineer turns used")
    _finish(ax, "Live ablation — identical task, models and engineer",
            "deepseek-v4-flash engineer, local qwen3:8b gates. n=1, temperature 0")
    return save(fig, "ablation.png")


def chart_literature():
    """What the literature reports, against what Gate 1 makes impossible."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    labels = [
        "Fabricated papers accepted\nby LLM reviewers (BadScientist)",
        "Nonexistent citations,\nMLR-Agent tasks (MLR-Bench)",
        "Best agent accuracy,\nCORE-Bench hardest level",
        "Best agent replication,\nPaperBench",
    ]
    values = [0.82, 0.50, 0.21, 0.21]
    bars = ax.barh(labels, values, 0.55, color=[ORANGE, ORANGE, BLUE, BLUE],
                   edgecolor=SURFACE, linewidth=2)
    for bar, v in zip(bars, values):
        ax.text(v + 0.015, bar.get_y() + bar.get_height() / 2, f"{v:.0%}",
                va="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=8.5)
    _finish(ax, "The measured landscape Gate 1 is placed into",
            "Orange: failure rates the layer targets. Blue: task-completion ceilings, for scale")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    return save(fig, "literature.png")


def chart_call_split():
    """Where a phase's model calls and tokens go."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    roles = ["engineer", "reward", "gate", "repair"]
    calls = [4, 5, 5, 1]
    chars = [37689, 14342, 10772, 1316]
    for ax, values, title, unit in (
        (ax1, calls, "model calls", ""),
        (ax2, chars, "prompt characters", ""),
    ):
        total = sum(values)
        colours = [MUTED, MUTED, BLUE, MUTED]
        bars = ax.bar(roles, values, 0.6, color=colours, edgecolor=SURFACE, linewidth=2)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + total * 0.02,
                    f"{100*v/total:.0f}%", ha="center", va="bottom", fontsize=8.5,
                    color=INK)
        ax.set_title(title, color=INK, fontsize=10, fontweight="bold", loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.set_ylim(0, max(values) * 1.22)
    fig.suptitle(
        "Gate 1 is a third of the calls and a sixth of the tokens",
        color=INK, fontsize=12, fontweight="bold", x=0.005, ha="left", y=1.06,
    )
    fig.text(0.005, 0.99, "Measured over five executions of one solver phase; "
             "blue is the gate", color=MUTED, fontsize=8.5, ha="left")
    return save(fig, "callsplit.png")


def chart_accuracy(run: dict | None):
    """How much of what each arm reported can actually be checked.

    Three quantities per arm: numbers reported, numbers checkable against an
    execution, numbers that reproduced when the accepted code was re-run. An arm
    that recorded nothing has nothing in the second and third bars, which is the
    finding rather than an omission.
    """
    if not run:
        return None
    arms = ["with Gate 1", "without Gate 1"]
    reported, checkable, reproduced = [], [], []
    for key in ("gated", "ungated"):
        v = run["arms"][key]["verification"]
        reported.append(len(v["reported"]))
        checkable.append(len(v["matched"]) + len(v["mismatched"]))
        reproduced.append(len(v["matched"]))

    x = range(len(arms))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    series = (
        (reported, MUTED, "reported", -width),
        (checkable, ORANGE, "checkable against an execution", 0.0),
        (reproduced, BLUE, "reproduced on re-execution", width),
    )
    for values, colour, label, off in series:
        bars = ax.bar([i + off for i in x], values, width, color=colour,
                      label=label, edgecolor=SURFACE, linewidth=2)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.08, str(v),
                    ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(arms, fontsize=10)
    ax.set_ylabel("numbers")
    top = max(reported + [1])
    ax.set_ylim(0, top * 1.35)
    _finish(ax, "How much of each arm's report can be checked",
            "One run, data-efficiency task. deepseek-v4-flash engineer, local "
            "qwen3:8b gates")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), ncol=3)
    return save(fig, "accuracy.png")


def main():
    bench_path = Path(
        "/tmp/claude-1000/-home-kesh/976cef29-0afd-479c-a716-6c557e07b6cb"
        "/scratchpad/bench2.json"
    )
    run_path = Path(
        "/home/kesh/AgentLaboratory-Gemini/ablation_runs/data_efficiency/ablation.json"
    )
    run = json.loads(run_path.read_text()) if run_path.exists() else None
    bench = json.loads(bench_path.read_text()) if bench_path.exists() else None
    if bench is None:
        print("  (no bench2.json — scanner chart will show the baseline only)")
    print("charts:")
    chart_check_inventory()
    chart_scanner(bench)
    chart_compression()
    chart_ablation(run)
    chart_literature()
    chart_call_split()
    if chart_accuracy(run) is None:
        print('  (no ablation.json — accuracy chart skipped)')


if __name__ == "__main__":
    main()
