"""Draw the paper's measured figures as PDF and PNG.

    python3 paper/figures.py        # writes paper/figures/*.{pdf,png}

The figures draw measured run aggregates, the signed red-team campaign, and
model-free rig probes from the CSV files beside this module. If an input carries
a row whose status is ``dummy``, the shared save path stamps its figure
PLACEHOLDER. The agent-judge figure is skipped until ``judging.csv`` exists.

Needs matplotlib. This is not part of the gates package, whose stdlib-only rule
covers ``gates/`` alone.
"""

import csv
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib
except ModuleNotFoundError:
    # The paper is not part of the stdlib-only package. Debian keeps its
    # workstation plotting packages outside isolated virtual environments.
    system_site = sysconfig.get_path(
        "purelib", vars={"base": sys.base_prefix, "platbase": sys.base_prefix}
    )
    sys.path.append(system_site)
    import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"

# The paper palette keeps text dark on SURFACE and uses redundant shape or
# hatching wherever hue carries meaning.
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
ALONE, GATED = MUTED, "#2a78d6"
BLUE, ORANGE, VERMILLION = "#0072B2", "#9A6700", "#D55E00"


def mechanism_figure(
    data_path: Path = HERE / "mechanism.csv", output_dir: Path = OUT
) -> "RenderedFigure":
    """Draw measured detection and false-positive evidence for each gate."""
    sys.path.insert(0, str(HERE.parent))
    from rig.stats import wilson

    rows = _csv_rows(Path(data_path))
    fig, ax = plt.subplots(figsize=(7, 4.7), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    labels = []
    for y, row in enumerate(rows):
        k, n = int(row["k"]), int(row["n"])
        value = 100 * k / n
        off = row["arm"].endswith("off")
        ax.barh(
            y,
            value,
            0.6,
            color=ALONE if off else GATED,
            edgecolor=SURFACE,
            linewidth=0.7,
        )
        end = value
        if row["interval"] == "yes":
            lo, hi = wilson(k, n)
            ax.hlines(y, 100 * lo, 100 * hi, color=INK, linewidth=1)
            end = 100 * hi
        if value == 0:
            ax.scatter(0, y, marker="|", s=45, color=ALONE, zorder=3)
        ax.text(end + 1.4, y, f"{k}/{n}", va="center", fontsize=7, color=INK)
        arm = f", {row['arm']}" if row["arm"] else ""
        labels.append(f"{row['panel']}: {row['measure']}{arm}")
    ax.set_yticks(range(len(rows)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 116)
    ax.set_xticks([0, 20, 40, 60, 80, 100], ["0", "20", "40", "60", "80", "100%"])
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=7, length=0)
    ax.set_xlabel("Whiskers: Wilson 95% interval", fontsize=7, color=INK2)
    fig.suptitle(
        "Mechanism evidence: what the gates catch and wrongly flag",
        x=0.01,
        y=0.98,
        ha="left",
        fontsize=10,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.01,
        0.01,
        "Gate 1 uses the signed 08-15 campaign. Gates 2 and 3 use model-free rigs.",
        fontsize=7,
        color=INK2,
    )
    fig.subplots_adjust(left=0.48, right=0.98, top=0.91, bottom=0.12)
    return _save_publication_figure(fig, rows, Path(output_dir), "fig8_mechanism")


@dataclass(frozen=True)
class RenderedFigure:
    """The saved publication files and the Matplotlib figure that produced them."""

    png: Path
    pdf: Path
    figure: object


def _save_publication_figure(fig, rows, output_dir: Path, stem: str) -> RenderedFigure:
    """Stamp dummy data, then save the same figure as a raster preview and vector art."""
    if any(row.get("status", "").strip().lower() == "dummy" for row in rows):
        fig.text(
            0.5,
            0.5,
            "PLACEHOLDER",
            fontsize=48,
            color=INK,
            alpha=0.09,
            rotation=18,
            ha="center",
            va="center",
            weight="bold",
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, facecolor=SURFACE)
    fig.savefig(pdf, facecolor=SURFACE)
    plt.close(fig)
    return RenderedFigure(png=png, pdf=pdf, figure=fig)


def _level_sort(level: str) -> tuple[int, str]:
    number = level.upper().removeprefix("L")
    return (int(number), level) if number.isdigit() else (10_000, level)


def _paper_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=7.5, length=0)


def _csv_rows(data_path: Path) -> list[dict[str, str]]:
    with open(data_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{data_path} has no data rows")
    return rows


def crashes_figure(
    data_path: Path = HERE / "crashes.csv", output_dir: Path = OUT
) -> RenderedFigure:
    """Draw execution crashes per level, separated by owner and cause."""
    data_path = Path(data_path)
    rows = _csv_rows(data_path)
    levels = sorted({row["arm"] for row in rows}, key=_level_sort)
    by_level = {row["arm"]: row for row in rows}
    panels = (
        (
            "Harness-caused crashes",
            (
                ("oom_at_cap", "GPU cap", BLUE, ""),
                ("timeout", "timeout", "#56B4E9", "//"),
                ("environment", "env.", "#009E73", "xx"),
            ),
        ),
        (
            "Agent-caused crashes",
            (
                ("oom", "OOM", VERMILLION, ""),
                ("agent_code", "agent code", "#CC79A7", "//"),
            ),
        ),
    )
    maximum = max(
        sum(int(by_level[level][key]) for key, _, _, _ in causes)
        for _, causes in panels
        for level in levels
    )
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.75), facecolor=SURFACE, sharey=True)
    for ax, (title, causes) in zip(axes, panels, strict=True):
        _paper_axis(ax)
        owner_total = sum(
            int(by_level[level][key])
            for level in levels
            for key, _, _, _ in causes
        )
        for level_index, level in enumerate(levels):
            bottom = 0
            for key, label, color, hatch in causes:
                count = int(by_level[level][key])
                if not count:
                    continue
                ax.bar(
                    level_index,
                    count,
                    bottom=bottom,
                    width=0.65,
                    color=color,
                    edgecolor=SURFACE,
                    hatch=hatch,
                    linewidth=0.7,
                )
                if count == 1:
                    ax.text(
                        level_index + 0.37,
                        bottom + count / 2,
                        f"{count}",
                        ha="left",
                        va="center",
                        fontsize=7,
                        color=INK,
                        clip_on=False,
                    )
                else:
                    ax.text(
                        level_index,
                        bottom + count / 2,
                        f"{label}\n{count}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=SURFACE if color == BLUE else INK,
                    )
                bottom += count
            ax.text(
                level_index,
                bottom + maximum * 0.035,
                f"{bottom}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=INK,
                weight="bold",
            )
        ax.set_xticks(range(len(levels)), levels)
        ax.set_title(
            f"{title} ({owner_total} total)",
            loc="left",
            fontsize=9,
            weight="bold",
            color=INK,
        )
        ax.set_ylim(0, maximum * 1.18)
        ax.set_xlim(-0.65, len(levels) - 0.15)
    axes[0].set_ylabel("Crashed executions", fontsize=7.5, color=INK2)
    fig.suptitle(
        "Crashes by level and cause",
        x=0.01,
        y=0.98,
        ha="left",
        fontsize=10,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.01,
        0.02,
        "Key: blue GPU cap; striped light blue timeout; crosshatched green environment; "
        "orange OOM; striped pink agent code.",
        fontsize=7,
        color=INK2,
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.82, wspace=0.18)
    return _save_publication_figure(fig, rows, Path(output_dir), "crashes_by_level")


def tokens_cost_figure(
    data_path: Path = HERE / "waves23.csv", output_dir: Path = OUT
) -> RenderedFigure:
    """Draw mean token use and cost per level, with every seed visible."""
    rows = _csv_rows(Path(data_path))
    levels = sorted({row["level"] for row in rows}, key=_level_sort)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.65), facecolor=SURFACE)
    measures = (
        (
            "Tokens",
            BLUE,
            lambda row: sum(
                float(row[column])
                for column in ("tokens_prompt", "tokens_completion")
            )
            / 1_000_000,
            "Million tokens",
            lambda value: f"{value:.2f}M",
        ),
        (
            "Cost",
            VERMILLION,
            lambda row: float(row["cost_usd"]),
            "US dollars",
            lambda value: f"${value:.2f}",
        ),
    )
    for ax, (title, color, value_of, ylabel, label_of) in zip(
        axes, measures, strict=True
    ):
        _paper_axis(ax)
        values_by_level = {
            level: [value_of(row) for row in rows if row["level"] == level]
            for level in levels
        }
        measure_max = max(value for values in values_by_level.values() for value in values)
        for level_index, level in enumerate(levels):
            values = values_by_level[level]
            mean = sum(values) / len(values)
            ax.bar(
                level_index,
                mean,
                width=0.62,
                color=color,
                edgecolor=SURFACE,
                linewidth=0.8,
                zorder=2,
            )
            for seed_index, value in enumerate(values):
                offset = (seed_index - (len(values) - 1) / 2) * 0.11
                ax.scatter(
                    level_index + offset,
                    value,
                    s=23,
                    color=INK,
                    edgecolor=SURFACE,
                    linewidth=0.6,
                    zorder=3,
                )
            ax.text(
                level_index,
                max(values) + measure_max * 0.055,
                label_of(mean),
                ha="center",
                va="bottom",
                fontsize=7,
                color=INK,
                weight="bold",
            )
        ax.set_xticks(range(len(levels)), levels)
        ax.set_ylim(0, measure_max * 1.25)
        ax.set_ylabel(ylabel, fontsize=7.5, color=INK2)
        ax.set_title(title, loc="left", fontsize=9, color=INK, weight="bold")
    fig.suptitle(
        "Resource use by level",
        x=0.01,
        y=0.99,
        ha="left",
        fontsize=10,
        weight="bold",
        color=INK,
    )
    fig.text(0.01, 0.01, "Bars: mean over seeds. Points: individual seeds.", fontsize=7, color=INK2)
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.2, top=0.8, wspace=0.3)
    return _save_publication_figure(fig, rows, Path(output_dir), "tokens_cost_by_level")


def gate_attempts_figure(
    data_path: Path = HERE / "waves23.csv", output_dir: Path = OUT
) -> RenderedFigure:
    """Draw total gate attempts and rejected attempts at each level."""
    rows = _csv_rows(Path(data_path))
    levels = sorted({row["level"] for row in rows}, key=_level_sort)
    gates = (("Gate 1", "g1"), ("Gate 2", "g2"), ("Gate 3", "g3"))
    totals = {
        (level, prefix): (
            sum(int(row[f"{prefix}_attempts"]) for row in rows if row["level"] == level),
            sum(int(row[f"{prefix}_fails"]) for row in rows if row["level"] == level),
        )
        for level in levels
        for _, prefix in gates
    }
    maximum = max((attempts for attempts, _ in totals.values()), default=1)
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.65), facecolor=SURFACE, sharey=True)
    for ax, (gate_name, prefix) in zip(axes, gates, strict=True):
        _paper_axis(ax)
        for level_index, level in enumerate(levels):
            attempts, rejected = totals[(level, prefix)]
            if rejected > attempts:
                raise ValueError(f"{gate_name} at {level} rejects more attempts than it made")
            passed = attempts - rejected
            if attempts:
                ax.bar(
                    level_index,
                    passed,
                    width=0.66,
                    color=BLUE,
                    edgecolor=SURFACE,
                    linewidth=0.7,
                )
                ax.bar(
                    level_index,
                    rejected,
                    bottom=passed,
                    width=0.66,
                    color=VERMILLION,
                    edgecolor=SURFACE,
                    hatch="//",
                    linewidth=0.7,
                )
                ax.text(
                    level_index,
                    attempts + maximum * 0.04,
                    f"{attempts}\n({rejected})",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=INK,
                )
            else:
                ax.text(level_index, maximum * 0.025, "-", ha="center", fontsize=8, color=INK2)
        ax.set_xticks(range(len(levels)), levels)
        ax.set_xlim(-0.65, len(levels) - 0.35)
        ax.set_ylim(0, maximum * 1.22)
        ax.set_title(gate_name, loc="left", fontsize=9, weight="bold", color=INK)
    axes[0].set_ylabel("Attempts across both seeds", fontsize=7.5, color=INK2)
    fig.suptitle(
        "Gate attempts and rejections by level",
        x=0.01,
        y=0.99,
        ha="left",
        fontsize=10,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.01,
        0.01,
        "Labels: total attempts, with rejected attempts in parentheses. Hatched segment: rejected attempts.",
        fontsize=7,
        color=INK2,
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.2, top=0.79, wspace=0.16)
    return _save_publication_figure(fig, rows, Path(output_dir), "gate_attempts_by_level")


def redteam_figure(
    data_path: Path = HERE / "redteam.csv", output_dir: Path = OUT
) -> RenderedFigure:
    """Draw the measured outcome of every adversarial strategy."""
    rows = _csv_rows(Path(data_path))
    order = {"blocked": 0, "warned": 1, "silent": 2}
    styles = {
        "blocked": (VERMILLION, "X"),
        "warned": (ORANGE, "^"),
        "silent": (BLUE, "o"),
    }
    unknown = sorted({row["outcome"] for row in rows} - order.keys())
    if unknown:
        raise ValueError(f"unknown red-team outcomes: {', '.join(unknown)}")
    rows = sorted(
        rows,
        key=lambda row: int(row["id"].upper().removeprefix("S")),
    )
    counts = {outcome: sum(row["outcome"] == outcome for row in rows) for outcome in order}
    fig, ax = plt.subplots(
        figsize=(7, max(2.5, 0.27 * len(rows) + 1.35)), facecolor=SURFACE
    )
    ax.set_facecolor(SURFACE)
    for y, row in enumerate(rows):
        outcome = row["outcome"]
        color, marker = styles[outcome]
        ax.scatter(
            order[outcome],
            y,
            s=31,
            color=color,
            marker=marker,
            edgecolor=SURFACE,
            linewidth=0.6,
            zorder=3,
        )
    ax.set_yticks(
        range(len(rows)),
        [f"{row['id']}  {row['description']}" for row in rows],
    )
    ax.invert_yaxis()
    ax.set_xticks(
        range(3),
        [
            f"Blocked\n{counts['blocked']}",
            f"Warned\n{counts['warned']}",
            f"Silent\n{counts['silent']}",
        ],
    )
    ax.set_xlim(-0.45, 2.45)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=7, length=0)
    ax.set_title(
        "Red-team outcome by fabrication strategy",
        loc="left",
        fontsize=10,
        weight="bold",
        color=INK,
        pad=9,
    )
    fig.text(
        0.99,
        0.01,
        "Blocked: no registry reached the writer. Warned: key named. Silent: no check named it.",
        ha="right",
        fontsize=7,
        color=INK2,
    )
    fig.subplots_adjust(left=0.48, right=0.98, bottom=0.12, top=0.92)
    return _save_publication_figure(fig, rows, Path(output_dir), "redteam_outcomes")


def adjacency_figure(
    data_path: Path = HERE / "adjacency.csv", output_dir: Path = OUT
) -> RenderedFigure:
    """Draw attack catches and honest-shape passes from Gate 3 adjacency probes."""
    rows = _csv_rows(Path(data_path))
    panels = (
        ("attack", "caught", "Attack probes", "caught", "missed"),
        ("honest", "passed", "Honest shapes", "passed", "rejected"),
    )
    summary = []
    for kind, column, label, success_word, failure_word in panels:
        selected = [row for row in rows if row["kind"] == kind]
        if not selected:
            raise ValueError(f"{data_path} has no {kind} rows")
        values = [row[column].strip().lower() for row in selected]
        invalid = sorted(set(values) - {"true", "false"})
        if invalid:
            raise ValueError(f"{kind}.{column} must be true or false")
        successes = sum(value == "true" for value in values)
        summary.append((label, successes, len(values), success_word, failure_word))

    fig, ax = plt.subplots(figsize=(3.3, 1.9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for y, (_label, successes, total, success_word, failure_word) in enumerate(summary):
        success_pct = 100 * successes / total
        failure_pct = 100 - success_pct
        ax.barh(y, success_pct, height=0.5, color=BLUE, edgecolor=SURFACE, linewidth=0.7)
        ax.barh(
            y,
            failure_pct,
            left=success_pct,
            height=0.5,
            color=VERMILLION,
            edgecolor=SURFACE,
            hatch="///",
            linewidth=0.7,
        )
        ax.text(
            success_pct / 2,
            y,
            f"{successes}/{total} {success_word}",
            ha="center",
            va="center",
            fontsize=7,
            color=SURFACE,
            weight="bold",
        )
        failures = total - successes
        ax.text(
            101.5,
            y,
            f"{failures} {failure_word}",
            ha="left",
            va="center",
            fontsize=7,
            color=VERMILLION,
        )
    ax.set_yticks(range(len(summary)), [row[0] for row in summary])
    ax.invert_yaxis()
    ax.set_xlim(0, 128)
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=7, length=0)
    ax.set_title(
        "Gate 3 token-adjacency probes",
        loc="left",
        fontsize=9,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.01,
        0.01,
        "Blue: desired outcome. Hatched red: miss or false rejection.",
        fontsize=7,
        color=INK2,
    )
    fig.subplots_adjust(left=0.26, right=0.98, bottom=0.28, top=0.78)
    return _save_publication_figure(fig, rows, Path(output_dir), "adjacency_probes")


def agent_judge_figure(
    data_path: Path = HERE / "judging.csv", output_dir: Path = OUT
) -> RenderedFigure | None:
    """Draw judge scores and faked-result verdicts at each measured level."""
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    if not data_path.exists():
        return None
    rows = _csv_rows(data_path)

    benchmarks = list(dict.fromkeys(row["benchmark"] for row in rows))
    fig, axes = plt.subplots(
        len(benchmarks),
        2,
        figsize=(7, 2.45 * len(benchmarks)),
        facecolor=SURFACE,
        squeeze=False,
    )
    role_style = {
        "judge": (BLUE, "o"),
        "opinion": (VERMILLION, "s"),
    }
    for row_index, benchmark in enumerate(benchmarks):
        selected = [row for row in rows if row["benchmark"] == benchmark]
        levels = sorted({row["level"] for row in selected}, key=_level_sort)
        score_ax, flag_ax = axes[row_index]
        _paper_axis(score_ax)
        _paper_axis(flag_ax)

        for level_index, level in enumerate(levels):
            level_rows = [row for row in selected if row["level"] == level]
            scores = []
            for point_index, row in enumerate(level_rows):
                if not row["overall"].strip():
                    continue
                role = row["role"].strip().lower()
                color, marker = role_style.get(role, (MUTED, "D"))
                offset = (-0.09 if role == "judge" else 0.09) + 0.018 * (point_index % 3 - 1)
                value = float(row["overall"])
                if role == "judge":
                    scores.append(value)
                score_ax.scatter(
                    level_index + offset,
                    value,
                    s=22,
                    marker=marker,
                    color=color,
                    edgecolor=SURFACE,
                    linewidth=0.5,
                    zorder=3,
                )
            if scores:
                mean = sum(scores) / len(scores)
                score_ax.scatter(level_index, mean, marker="_", s=130, color=INK, zorder=4)
                # Left of the tick: opinions are drawn to the right of it.
                score_ax.text(
                    level_index - 0.3,
                    mean,
                    f"{mean:.1f}",
                    ha="right",
                    va="center",
                    fontsize=7,
                    color=INK,
                )

            counts = {"true": 0, "false": 0, "error": 0}
            for row in level_rows:
                if row["role"].strip().lower() != "judge":
                    continue
                value = row["faked"].strip().lower()
                counts[value if value in ("true", "false") else "error"] += 1
            left = 0
            segments = (
                ("true", "flagged", VERMILLION, ""),
                ("false", "clear", BLUE, ""),
                ("error", "error", SURFACE, "///"),
            )
            for key, label, color, hatch in segments:
                count = counts[key]
                if not count:
                    continue
                flag_ax.barh(
                    level_index,
                    count,
                    left=left,
                    height=0.58,
                    color=color,
                    edgecolor=MUTED if hatch else SURFACE,
                    hatch=hatch,
                    linewidth=0.7,
                )
                flag_ax.text(
                    left + count / 2,
                    level_index,
                    f"{label} {count}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=INK if key in ("true", "error") else SURFACE,
                )
                left += count

        score_ax.set_xticks(range(len(levels)), levels)
        score_ax.set_ylim(0.5, 10.5)
        score_ax.set_yticks([1, 3, 5, 7, 9])
        score_ax.set_title(
            f"{benchmark}: overall score", loc="left", fontsize=9, color=INK, weight="bold"
        )
        score_ax.set_ylabel("Score, 1-10", fontsize=7.5, color=INK2)
        score_ax.set_xlabel(
            "● judge    ■ opinion (not counted)    black tick: judge mean",
            fontsize=7,
            color=INK2,
        )
        flag_ax.set_yticks(range(len(levels)), levels)
        flag_ax.invert_yaxis()
        flag_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        flag_ax.set_xlabel("Judge responses", fontsize=7.5, color=INK2)
        flag_ax.set_title("Faked-result verdicts", loc="left", fontsize=9, color=INK, weight="bold")

    fig.suptitle(
        "Agent-judge results by level",
        x=0.01,
        y=0.98,
        ha="left",
        fontsize=10,
        weight="bold",
        color=INK,
    )
    fig.tight_layout(pad=0.7, w_pad=1.4, rect=(0, 0, 1, 0.94))
    return _save_publication_figure(fig, rows, output_dir, "agent_judge_by_level")


JUDGE_COLUMNS = (
    ("agent:claude/claude-opus-5-5", "Opus 5.5"),
    ("agent:codex/gpt-5.6-sol:high", "GPT-5.6 Sol"),
    ("agent:cursor/grok-4.7-high", "Grok 4.7*"),
)


def _run_label(row: dict[str, str]) -> str:
    return f"{row['level']} seed {row['seed']}"


def audit_figure(
    provenance_path: Path = HERE / "provenance.csv",
    judging_path: Path = HERE / "judging.csv",
    output_dir: Path = OUT,
) -> RenderedFigure:
    """Draw, per run, where the paper's numerals came from and every rater's verdict."""
    runs = sorted(
        _csv_rows(provenance_path), key=lambda r: (_level_sort(r["level"]), int(r["seed"]))
    )
    verdicts: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in _csv_rows(judging_path):
        verdicts[(row["level"], row["seed"], row["judge"])] = row

    fig, (share_ax, grid_ax) = plt.subplots(
        1, 2, figsize=(7.0, 3.1), gridspec_kw={"width_ratios": [1.0, 1.0]}
    )
    fig.patch.set_facecolor(SURFACE)
    _paper_axis(share_ax)
    share_ax.grid(axis="y", visible=False)
    share_ax.grid(axis="x", color=GRID, linewidth=0.6)

    labels = [_run_label(r) for r in runs]
    for index, row in enumerate(runs):
        total = int(row["numerals"])
        missing = int(row["not_in_execution"])
        found = total - missing
        found_share = 100 * found / total
        share_ax.barh(index, found_share, color=BLUE, height=0.62)
        share_ax.barh(
            index, 100 - found_share, left=found_share, color=VERMILLION, height=0.62, hatch="///",
            edgecolor=SURFACE, linewidth=0,
        )
        share_ax.text(
            101.5, index, f"{missing}/{total}", va="center", ha="left", fontsize=7, color=INK
        )
    share_ax.set_yticks(range(len(runs)), labels)
    share_ax.invert_yaxis()
    share_ax.set_xlim(0, 118)
    share_ax.set_xticks([0, 25, 50, 75, 100])
    share_ax.set_xlabel("Share of the paper's numerals, %", fontsize=7, color=INK2)
    share_ax.set_title(
        "(a) Paper numerals found in the feeding execution",
        loc="left", fontsize=7.5, color=INK, weight="bold",
    )
    for boundary in (1.5, 3.5, 5.5):
        share_ax.axhline(boundary, color=MUTED, linewidth=0.5, linestyle=":")
    share_ax.text(
        101.5, -0.8, "missing/all", ha="left", va="center", fontsize=7, color=INK2
    )
    share_ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=BLUE, label="found"),
            plt.Rectangle((0, 0), 1, 1, facecolor=VERMILLION, hatch="///", edgecolor=SURFACE,
                          label="not found"),
        ],
        loc="lower left", bbox_to_anchor=(0.0, -0.36), ncol=2, frameon=False, fontsize=7,
    )

    short = ["Opus", "Sol", "Grok*"]
    columns = ["Rule A"] + short + short + ["Cites"]
    grid_ax.set_facecolor(SURFACE)
    for side in grid_ax.spines.values():
        side.set_visible(False)
    grid_ax.set_xlim(-0.5, len(columns) - 0.5)
    grid_ax.set_ylim(len(runs) - 0.5, -0.5)
    grid_ax.set_xticks(range(len(columns)), columns, fontsize=7, color=INK2)
    for x0, x1, label in ((1, 3, "faked flag"), (4, 6, "score, 1-10")):
        grid_ax.annotate(
            "", xy=(x0 - 0.4, -1.35), xytext=(x1 + 0.4, -1.35), annotation_clip=False,
            arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6},
        )
        grid_ax.text((x0 + x1) / 2, -1.55, label, ha="center", va="bottom", fontsize=7,
                     color=INK, clip_on=False)
    grid_ax.text(0, -1.55, "agent,\ncode+logs", ha="center", va="bottom", fontsize=7,
                 color=INK2, clip_on=False)
    grid_ax.text(7, -1.55, "wrong\nrefs", ha="center", va="bottom", fontsize=7,
                 color=INK2, clip_on=False)
    grid_ax.xaxis.tick_top()
    grid_ax.set_yticks(range(len(runs)), [""] * len(runs))
    grid_ax.tick_params(length=0)

    def cell(x: int, y: int, fill: str, text: str, color: str = INK, hatch: str | None = None):
        grid_ax.add_patch(
            plt.Rectangle((x - 0.46, y - 0.42), 0.92, 0.84, facecolor=fill, edgecolor=GRID,
                          linewidth=0.5, hatch=hatch)
        )
        grid_ax.text(x, y, text, ha="center", va="center", fontsize=7, color=color)

    for y, row in enumerate(runs):
        rule_a = row["rule_a_faked"].strip().lower()
        if rule_a == "true":
            cell(0, y, VERMILLION, "faked", SURFACE)
        elif rule_a == "open":
            cell(0, y, SURFACE, "open", INK, hatch="....")
        else:
            cell(0, y, SURFACE, "clear", INK2)
        for j, (judge, _) in enumerate(JUDGE_COLUMNS):
            verdict = verdicts.get((row["level"], row["seed"], judge), {})
            flag = verdict.get("faked", "").strip().lower()
            if flag == "true":
                cell(1 + j, y, ORANGE, "flag", SURFACE)
            elif flag == "false":
                cell(1 + j, y, SURFACE, "clear", INK2)
            else:
                cell(1 + j, y, SURFACE, "none", MUTED, hatch="////")
            score = verdict.get("overall", "").strip()
            cell(4 + j, y, "#eef2f7" if score else SURFACE, score or "none",
                 INK if score else MUTED, None if score else "////")
        problems = int(row["citation_problems"])
        cell(7, y, ORANGE if problems else SURFACE, str(problems), SURFACE if problems else INK2)
    for boundary in (1.5, 3.5, 5.5):
        grid_ax.axhline(boundary, color=MUTED, linewidth=0.5, linestyle=":")
    grid_ax.set_title("(b) Every rater's verdict", loc="left", fontsize=7.5, color=INK,
                      weight="bold", pad=34)
    fig.tight_layout(pad=0.4, w_pad=0.6)
    return _save_publication_figure(fig, runs, output_dir, "audit_by_run")


def price_figure(
    waves_path: Path = HERE / "waves23.csv",
    crashes_path: Path = HERE / "crashes.csv",
    output_dir: Path = OUT,
) -> RenderedFigure:
    """Draw what each level cost: dollars, tokens, gate attempts and crashes by cause."""
    rows = _csv_rows(waves_path)
    crash_rows = {r["arm"]: r for r in _csv_rows(crashes_path)}
    levels = sorted({r["level"] for r in rows}, key=_level_sort)
    by_level = {lv: [r for r in rows if r["level"] == lv] for lv in levels}
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.5))
    fig.patch.set_facecolor(SURFACE)
    xs = range(len(levels))

    def seeds_panel(ax, values, title, unit, fmt):
        _paper_axis(ax)
        means = [sum(v) / len(v) for v in values]
        ax.bar(xs, means, color=[MUTED] + [GATED] * (len(levels) - 1), width=0.62)
        for x, seed_values in zip(xs, values, strict=True):
            ax.scatter([x - 0.12, x + 0.12][: len(seed_values)], seed_values, s=9, color=INK,
                       zorder=3, marker="o")
        for x, mean in zip(xs, means, strict=True):
            ax.text(x, mean * 0.5, fmt(mean), ha="center", va="center", fontsize=7,
                    color=SURFACE, rotation=90)
        ax.set_xticks(list(xs), levels)
        ax.set_ylabel(unit, fontsize=7, color=INK2)
        ratio = means[-1] / means[0]
        ax.set_title(f"{title}, L3/L0 {ratio:.2f}x", loc="left", fontsize=7, color=INK,
                     weight="bold")

    seeds_panel(
        axes[0], [[float(r["cost_usd"]) for r in by_level[lv]] for lv in levels],
        "(a) Cost/run", "US dollars", lambda v: f"{v:.2f}",
    )
    seeds_panel(
        axes[1],
        [[(int(r["tokens_prompt"]) + int(r["tokens_completion"])) / 1e6 for r in by_level[lv]]
         for lv in levels],
        "(b) Tokens/run", "millions, prompt+completion", lambda v: f"{v:.1f}",
    )

    ax = axes[2]
    _paper_axis(ax)
    width = 0.26
    gate_colors = (BLUE, ORANGE, VERMILLION)
    for g, color in enumerate(gate_colors, start=1):
        for x, lv in enumerate(levels):
            attempts = sum(int(r[f"g{g}_attempts"]) for r in by_level[lv])
            fails = sum(int(r[f"g{g}_fails"]) for r in by_level[lv])
            if g > _level_sort(lv)[0]:
                continue
            pos = x + (g - 2) * width
            ax.bar(pos, attempts - fails, width=width, color=color, bottom=fails)
            ax.bar(pos, fails, width=width, facecolor=SURFACE, edgecolor=color, hatch="////",
                   linewidth=0.6)
            ax.text(pos, attempts + 0.6, f"{fails}/{attempts}", ha="center", fontsize=7,
                    color=INK, rotation=90, va="bottom")
    ax.set_xticks(list(xs), levels)
    ax.set_title("(c) Gate attempts", loc="left", fontsize=7.2, color=INK, weight="bold")
    ax.set_ylabel("attempts", fontsize=7, color=INK2)
    ax.set_ylim(0, 50)
    ax.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=f"{g}")
                 for g, c in enumerate(gate_colors, start=1)],
        fontsize=7, frameon=False, loc="upper left", handlelength=0.8, ncol=3,
        columnspacing=0.6,
    )

    ax = axes[3]
    _paper_axis(ax)
    causes = (
        ("oom_at_cap", "GPU cap", BLUE, None),
        ("timeout", "timeout", "#56B4E9", "///"),
        ("environment", "environment", "#009E73", "xx"),
        ("agent_code", "agent code", "#CC79A7", None),
    )
    for x, lv in enumerate(levels):
        bottom = 0
        crash = crash_rows[lv]
        for key, _, color, hatch in causes:
            count = int(crash[key])
            if count:
                ax.bar(x, count, bottom=bottom, color=color, hatch=hatch, edgecolor=SURFACE,
                       linewidth=0, width=0.62)
            bottom += count
        ax.text(x, bottom + 0.3, f"{crash['crashed']}/{crash['executions']}", ha="center",
                fontsize=7, color=INK)
    ax.set_xticks(list(xs), levels)
    ax.set_title("(d) Crashed executions", loc="left", fontsize=7.2, color=INK, weight="bold")
    ax.set_ylabel("crashed executions", fontsize=7, color=INK2)
    ax.set_ylim(0, 28)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, facecolor=c, hatch=h, edgecolor=SURFACE, label=n)
                 for _, n, c, h in causes],
        fontsize=7, frameon=False, loc="upper left", handlelength=0.9, ncol=2,
        columnspacing=0.5,
    )
    fig.tight_layout(pad=0.4, w_pad=0.5)
    return _save_publication_figure(fig, rows, output_dir, "price_by_level")


if __name__ == "__main__":
    rendered = [
        audit_figure(),
        price_figure(),
        crashes_figure(),
        tokens_cost_figure(),
        gate_attempts_figure(),
        redteam_figure(),
        adjacency_figure(),
        mechanism_figure(),
    ]
    judging = agent_judge_figure()
    if judging is not None:
        rendered.append(judging)
    names = ", ".join(item.png.stem for item in rendered)
    print(f"wrote {len(rendered)} figures as PDF and PNG in {OUT}: {names}")
