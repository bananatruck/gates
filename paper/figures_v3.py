"""Draw the v3 draft's figures under the dataviz method, as PDF and PNG.

    python3 paper/figures_v3.py        # writes paper/figures/v3_*.{pdf,png}

Every value comes from the CSVs beside this module. The palette is the dataviz
skill's reference instance, validated with its `validate_palette.js` on
2026-10-01: the gates use categorical slots 1-3 (all-pairs PASS), the crash
causes slots 1-4 (adjacent PASS), the levels an ordinal blue ramp (PASS), and
verdict states the fixed status palette, always with a text label beside or
inside the mark. Aqua and yellow sit below 3:1 on the surface, so every mark
in those hues is direct-labelled and its numbers are in the paper's tables.

These are print figures: no hover layer and no dark mode. The 45-degree
texture marks rejected or missing shares, which the method allows for print.

Needs matplotlib. Not part of the stdlib-only `gates/` package.
"""

import csv
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib
except ModuleNotFoundError:
    system_site = sysconfig.get_path(
        "purelib", vars={"base": sys.base_prefix, "platbase": sys.base_prefix}
    )
    sys.path.append(system_site)
    import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"

# Chart chrome and ink (dataviz reference palette, light mode).
SURFACE, PLANE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
# Categorical slots in fixed order.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GATE_COLOR = {1: BLUE, 2: ORANGE, 3: AQUA}
# Ordinal ramp for the cumulative levels (blue 250, 400, 550, 700).
LEVEL_COLOR = {"L0": "#86b6ef", "L1": "#3987e5", "L2": "#1c5cab", "L3": "#0d366b"}
# Status palette: fixed, never themed, always with a label.
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
# Tone-on-tone steps for texture and washes.
BLUE_250, BLUE_100 = "#86b6ef", "#cde2fb"
NEUTRAL = "#f0efec"

FONT = 7.0  # the smallest text on any figure, in points
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
        "font.size": FONT,
        "hatch.linewidth": 0.6,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.6,
    }
)


@dataclass(frozen=True)
class RenderedFigure:
    png: Path
    pdf: Path
    figure: object


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return rows


def _save(fig, output_dir: Path, stem: str) -> RenderedFigure:
    output_dir.mkdir(parents=True, exist_ok=True)
    png, pdf = output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, facecolor=SURFACE)
    fig.savefig(pdf, facecolor=SURFACE, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    return RenderedFigure(png=png, pdf=pdf, figure=fig)


def _axis(ax, grid_axis: str = "y") -> None:
    """Recessive chrome: hairline solid grid, one baseline, no box."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.5, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    if grid_axis == "y":
        ax.spines["bottom"].set_visible(True)
    else:
        ax.spines["left"].set_visible(True)
    ax.tick_params(colors=INK2, labelsize=FONT, length=0)


def _title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=7.5, color=INK, weight="bold", pad=6)


def _rounded_bar(ax, x0, x1, y0, y1, color, *, horizontal=False, radius_pt=1.6,
                 hatch=None, hatch_color=None, rounded=True):
    """A bar with a rounded data end and a square baseline end.

    Axis limits must be final before calling, because the corner radius is
    converted from points to data units through the current transform.
    """
    fig = ax.figure
    px = radius_pt * fig.dpi / 72.0
    inv = ax.transData.inverted()
    ox, oy = ax.transData.transform((0, 0))
    rx = abs(inv.transform((ox + px, oy))[0] - inv.transform((ox, oy))[0])
    ry = abs(inv.transform((ox, oy + px))[1] - inv.transform((ox, oy))[1])
    rx = min(rx, abs(x1 - x0) / 2)
    ry = min(ry, abs(y1 - y0) / 2)
    if not rounded:
        rx = ry = 0
    if horizontal:  # data end is x1
        verts = [(x0, y0), (x1 - rx, y0), (x1, y0), (x1, y0 + ry), (x1, y1 - ry), (x1, y1),
                 (x1 - rx, y1), (x0, y1), (x0, y0)]
    else:  # data end is y1
        verts = [(x0, y0), (x0, y1 - ry), (x0, y1), (x0 + rx, y1), (x1 - rx, y1), (x1, y1),
                 (x1, y1 - ry), (x1, y0), (x0, y0)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO,
             MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CLOSEPOLY]
    patch = PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", linewidth=0)
    ax.add_patch(patch)
    if hatch:
        ax.add_patch(PathPatch(MplPath(verts, codes), facecolor="none",
                               edgecolor=hatch_color or SURFACE, hatch=hatch, linewidth=0))
    return patch


def _ink_on(fill: str) -> str:
    """White or ink, whichever clears contrast on the fill."""
    r, g, b = (int(fill[i:i + 2], 16) / 255 for i in (1, 3, 5))

    def lin(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    return INK if (lum + 0.05) / 0.05 > 1.05 / (lum + 0.05) else "#ffffff"


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        raise ValueError("no trials")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# --------------------------------------------------------------------------- #
# Figure 1: key results, a KPI row of stat tiles
# --------------------------------------------------------------------------- #

def key_results_figure(
    provenance_path: Path = HERE / "provenance.csv",
    redteam_path: Path = HERE / "redteam.csv",
    adjacency_path: Path = HERE / "adjacency.csv",
    waves_path: Path = HERE / "waves23.csv",
    output_dir: Path = OUT,
) -> RenderedFigure:
    prov = _rows(provenance_path)
    l0 = [r for r in prov if r["level"] == "L0"]
    l13 = [r for r in prov if r["level"] in ("L1", "L3")]
    l3 = [r for r in prov if r["level"] == "L3"]
    red = _rows(redteam_path)
    adj = [r for r in _rows(adjacency_path) if r["kind"] == "attack"]
    waves = _rows(waves_path)

    def cost(level):
        return sum(float(r["cost_usd"]) for r in waves if r["level"] == level)

    caught = sum(r["outcome"] in ("blocked", "warned") for r in red)
    blocked = sum(r["outcome"] == "blocked" for r in red)
    warned = sum(r["outcome"] == "warned" for r in red)
    tiles = [
        ("Ungated papers, faked",
         f"{sum(r['rule_a_faked'] == 'true' for r in l0)}/{len(l0)}",
         "level 0, agent trace"),
        ("Gated papers, faked",
         f"{sum(r['rule_a_faked'] == 'true' for r in l13)}/{len(l13)}",
         "levels 1, 3; L2 open"),
        ("Untraced numerals, L3",
         f"{sum(int(r['not_in_execution']) for r in l3)}/{sum(int(r['numerals']) for r in l3)}",
         "two papers"),
        ("Fabrications caught",
         f"{caught}/{len(red)}",
         f"{blocked} blocked, {warned} warned"),
        ("Adjacency attacks caught",
         f"{sum(r['caught'] == 'true' for r in adj)}/{len(adj)}",
         "Gate 3 probes"),
        ("Cost, L3 over L0",
         f"{cost('L3') / cost('L0'):.2f}×",
         "per run"),
    ]
    fig = plt.figure(figsize=(7.0, 1.05))
    fig.patch.set_facecolor(SURFACE)
    n = len(tiles)
    gap = 0.012
    width = (1 - gap * (n + 1)) / n
    for i, (label, value, context) in enumerate(tiles):
        x = gap + i * (width + gap)
        ax = fig.add_axes((x, 0.06, width, 0.88))
        ax.set_axis_off()
        ax.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0,rounding_size=0.06",
                                    transform=ax.transAxes, facecolor=PLANE,
                                    edgecolor=(11 / 255, 11 / 255, 11 / 255, 0.10), linewidth=0.6))
        ax.text(0.08, 0.80, label, transform=ax.transAxes, fontsize=FONT, color=INK2,
                va="top", ha="left", wrap=True)
        ax.text(0.08, 0.40, value, transform=ax.transAxes, fontsize=15, color=INK,
                weight="bold", va="center", ha="left")
        ax.text(0.08, 0.12, context, transform=ax.transAxes, fontsize=FONT, color=MUTED,
                va="center", ha="left")
    # Wrap labels to the tile width.
    for ax in fig.axes:
        label = ax.texts[0]
        words, lines, line = label.get_text().split(), [], ""
        for word in words:
            trial = f"{line} {word}".strip()
            if len(trial) > 18:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
        label.set_text("\n".join(lines))
    return _save(fig, output_dir, "v3_key_results")


# --------------------------------------------------------------------------- #
# Figure 3: mechanism evidence as a dot-and-interval plot
# --------------------------------------------------------------------------- #

# Gate 1's delivery row counts attempts, not values: the 15 August campaign
# required 4 values per attempt, so 40 of 40 values are 10 of 10 attempts, and
# the values within an attempt are not independent trials.
VALUES_PER_ATTEMPT = 4


def mechanism_figure_v3(
    mechanism_path: Path = HERE / "mechanism.csv",
    adjacency_path: Path = HERE / "adjacency.csv",
    output_dir: Path = OUT,
) -> RenderedFigure:
    mech = {r["id"]: r for r in _rows(mechanism_path)}
    adj = _rows(adjacency_path)
    attacks = [r for r in adj if r["kind"] == "attack"]
    honest = [r for r in adj if r["kind"] == "honest"]

    def kn(row_id, unit=1):
        r = mech[row_id]
        return int(r["k"]) // unit, int(r["n"]) // unit

    caught = [
        (1, "Values reach writer, Gate 1 on", *kn("g1_delivered_on", VALUES_PER_ATTEMPT), True, False),
        (1, "  same, Gate 1 off", *kn("g1_delivered_off", VALUES_PER_ATTEMPT), True, True),
        (1, "Claims traceable, Gate 1 on", *kn("g1_traced_on"), True, False),
        (1, "  same, Gate 1 off", *kn("g1_traced_off"), True, True),
        (2, "Defective registries rejected", *kn("g2a_detect"), True, False),
        (2, "Plan divergences caught", *kn("g2b_detect"), True, False),
        (2, "Unverifiable fields flagged", *kn("g2b_trace"), True, False),
        (3, "Typed literals detected", *kn("g3_detect"), False, False),
        (3, "Adjacency attacks caught", sum(r["caught"] == "true" for r in attacks), len(attacks), True, False),
    ]
    wrong = [
        (2, "Clean registries rejected", *kn("g2a_fpr"), True, False),
        (2, "Divergence false alarms", *kn("g2b_fpr"), True, False),
        (2, "Traceability false alarms", *kn("g2b_trace_fpr"), True, False),
        (3, "Honest token shapes rejected", sum(r["passed"] != "true" for r in honest), len(honest), True, False),
    ]
    fig, axes = plt.subplots(
        2, 1, figsize=(3.33, 3.05), gridspec_kw={"height_ratios": [len(caught), len(wrong)]},
        sharex=True,
    )
    fig.patch.set_facecolor(SURFACE)
    for ax, rows, title in ((axes[0], caught, "(a) Caught or delivered"),
                            (axes[1], wrong, "(b) Wrongly flagged")):
        _axis(ax, "x")
        ax.set_xlim(0, 100)
        ax.set_ylim(len(rows) - 0.5, -0.6)
        for y, (gate, _label, k, n, interval, comparison) in enumerate(rows):
            color = MUTED if comparison else GATE_COLOR[gate]
            share = 100 * k / n
            if interval:
                lo, hi = wilson(k, n)
                ax.plot([100 * lo, 100 * hi], [y, y], color=color, linewidth=2,
                        solid_capstyle="round", alpha=0.45)
            ax.scatter([share], [y], s=26, color=color, edgecolors=SURFACE, linewidths=1.2,
                       zorder=3, marker="o")
            ax.text(105, y, f"{k}/{n}", va="center", ha="left", fontsize=FONT, color=INK,
                    clip_on=False)
        ax.set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=FONT, color=INK2)
        _title(ax, title)
    axes[1].set_xticks([0, 25, 50, 75, 100])
    axes[1].set_xlabel("Share, %, with Wilson 95% interval", fontsize=FONT, color=INK2)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=5,
                          markerfacecolor=GATE_COLOR[g], markeredgecolor=SURFACE,
                          label=f"Gate {g}") for g in (1, 2, 3)]
    handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=5,
                              markerfacecolor=MUTED, markeredgecolor=SURFACE,
                              label="gate off"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=FONT,
               handletextpad=0.2, columnspacing=0.8, bbox_to_anchor=(0.55, -0.01))
    fig.tight_layout(pad=0.3, h_pad=0.6, rect=(0, 0.06, 0.95, 1))
    return _save(fig, output_dir, "v3_mechanism")


# --------------------------------------------------------------------------- #
# Figure 4: the red team as status cards
# --------------------------------------------------------------------------- #

def redteam_figure_v3(redteam_path: Path = HERE / "redteam.csv",
                      output_dir: Path = OUT) -> RenderedFigure:
    rows = _rows(redteam_path)
    groups = [
        ("blocked", "Blocked", GOOD, "no registry reached the writer"),
        ("warned", "Warned", WARNING, "a check named the planted key"),
        ("silent", "Silent", CRITICAL, "no check named it"),
    ]
    tallest = max(sum(r["outcome"] == key for r in rows) for key, *_ in groups)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 0.42 + 0.155 * tallest),
                             gridspec_kw={"width_ratios": [1.0, 0.95, 1.12]})
    fig.patch.set_facecolor(SURFACE)
    for ax, (key, name, color, meaning) in zip(axes, groups, strict=True):
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(tallest + 0.2, -1.5)
        members = [r for r in rows if r["outcome"] == key]
        ax.add_patch(FancyBboxPatch((0.0, -1.35), 0.055, 0.7, boxstyle="round,pad=0,rounding_size=0.02",
                                    facecolor=color, edgecolor="none"))
        ax.text(0.08, -1.0, f"{name}  {len(members)}", fontsize=8, weight="bold", color=INK,
                va="center")
        ax.text(0.08, -0.35, meaning, fontsize=FONT, color=INK2, va="center")
        for y, r in enumerate(members):
            ax.add_patch(Rectangle((0.0, y + 0.12), 0.012, 0.76, facecolor=color, edgecolor="none"))
            ax.text(0.03, y + 0.5, r["id"], fontsize=FONT, color=MUTED, va="center")
            ax.text(0.13, y + 0.5, r["description"], fontsize=FONT, color=INK, va="center")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02, wspace=0.08)
    return _save(fig, output_dir, "v3_redteam")


# --------------------------------------------------------------------------- #
# Figure 5: the per-run audit
# --------------------------------------------------------------------------- #

JUDGES = (
    ("agent:claude/claude-opus-5-5", "Opus"),
    ("agent:codex/gpt-5.6-sol:high", "Sol"),
    ("agent:cursor/grok-4.7-high", "Grok*"),
)


def _level_key(level: str) -> int:
    return int(level.upper().removeprefix("L"))


def audit_figure_v3(
    provenance_path: Path = HERE / "provenance.csv",
    judging_path: Path = HERE / "judging.csv",
    output_dir: Path = OUT,
) -> RenderedFigure:
    runs = sorted(_rows(provenance_path), key=lambda r: (_level_key(r["level"]), int(r["seed"])))
    verdicts = {(r["level"], r["seed"], r["judge"]): r for r in _rows(judging_path)}
    fig, (bar_ax, grid_ax) = plt.subplots(
        1, 2, figsize=(7.0, 2.55), gridspec_kw={"width_ratios": [1.0, 1.08]}
    )
    fig.patch.set_facecolor(SURFACE)

    _axis(bar_ax, "x")
    bar_ax.set_xlim(0, 100)
    bar_ax.set_ylim(len(runs) - 0.5, -0.5)
    half = 0.30
    for y, row in enumerate(runs):
        total, missing = int(row["numerals"]), int(row["not_in_execution"])
        found = 100 * (total - missing) / total
        _rounded_bar(bar_ax, 0, found, y - half, y + half, BLUE_250, horizontal=True,
                     rounded=missing == 0)
        if missing:
            gap = 0.35
            _rounded_bar(bar_ax, found + gap, 100, y - half, y + half, CRITICAL,
                         horizontal=True, hatch="////", hatch_color="#f3b3b3")
        bar_ax.text(101.5, y, f"{missing}/{total}", va="center", ha="left", fontsize=FONT,
                    color=INK, clip_on=False)
        bar_ax.add_patch(Rectangle((-3.4, y - half), 2.0, 2 * half,
                                   facecolor=LEVEL_COLOR[row["level"]], edgecolor="none",
                                   clip_on=False))
    for boundary in (1.5, 3.5, 5.5):
        bar_ax.axhline(boundary, color=GRID, linewidth=0.6)
    bar_ax.set_yticks(range(len(runs)), [f"{r['level']} seed {r['seed']}" for r in runs])
    bar_ax.tick_params(axis="y", pad=9)
    bar_ax.set_xticks([0, 25, 50, 75, 100])
    bar_ax.set_xlabel("Share of the paper's numerals, %", fontsize=FONT, color=INK2)
    _title(bar_ax, "(a) Numerals traced to the feeding run")
    bar_ax.text(101.5, -0.85, "missing/all", fontsize=FONT, color=INK2, ha="left", clip_on=False)
    bar_ax.legend(
        handles=[Rectangle((0, 0), 1, 1, facecolor=BLUE_250, label="found"),
                 Rectangle((0, 0), 1, 1, facecolor=CRITICAL, hatch="////",
                           edgecolor="#f3b3b3", label="not found")],
        loc="upper left", bbox_to_anchor=(0.0, -0.17), ncol=2, frameon=False, fontsize=FONT,
    )

    columns = ["Rule A"] + [n for _, n in JUDGES] + [n for _, n in JUDGES] + ["Refs"]
    grid_ax.set_facecolor(SURFACE)
    grid_ax.set_axis_off()
    grid_ax.set_xlim(-0.5, len(columns) - 0.5)
    grid_ax.set_ylim(len(runs) - 0.5, -2.3)

    def cell(x, y, fill, text, hatch=None):
        grid_ax.add_patch(FancyBboxPatch((x - 0.44, y - 0.38), 0.88, 0.76,
                                         boxstyle="round,pad=0,rounding_size=0.08",
                                         facecolor=fill, edgecolor="none", hatch=hatch))
        grid_ax.text(x, y, text, ha="center", va="center", fontsize=FONT,
                     color=_ink_on(fill) if fill not in (SURFACE, PLANE, NEUTRAL) else INK2)

    for x, name in enumerate(columns):
        grid_ax.text(x, -0.95, name, ha="center", va="center", fontsize=FONT, color=INK2)
    for x0, x1, label in ((0, 0, "trace"), (1, 3, "faked-result flag"), (4, 6, "score, 1-10"),
                          (7, 7, "wrong")):
        grid_ax.text((x0 + x1) / 2, -1.75, label, ha="center", va="center", fontsize=FONT,
                     color=INK)
        if x1 > x0:
            grid_ax.plot([x0 - 0.4, x1 + 0.4], [-1.42, -1.42], color=BASELINE, linewidth=0.6)
    _title(grid_ax, "(b) Every rater's verdict")
    for y, row in enumerate(runs):
        rule = row["rule_a_faked"].strip().lower()
        if rule == "true":
            cell(0, y, CRITICAL, "faked")
        elif rule == "open":
            cell(0, y, WARNING, "open")
        else:
            cell(0, y, NEUTRAL, "clear")
        for j, (judge, _) in enumerate(JUDGES):
            v = verdicts.get((row["level"], row["seed"], judge), {})
            flag = v.get("faked", "").strip().lower()
            if flag == "true":
                cell(1 + j, y, SERIOUS, "flag")
            elif flag == "false":
                cell(1 + j, y, NEUTRAL, "clear")
            else:
                cell(1 + j, y, SURFACE, "none", hatch="////")
            score = v.get("overall", "").strip()
            cell(4 + j, y, PLANE if score else SURFACE, score or "none")
        problems = int(row["citation_problems"])
        cell(7, y, SERIOUS if problems else NEUTRAL, str(problems))
    for boundary in (1.5, 3.5, 5.5):
        grid_ax.plot([-0.5, len(columns) - 0.5], [boundary, boundary], color=GRID, linewidth=0.6)
    fig.tight_layout(pad=0.4, w_pad=1.0)
    return _save(fig, output_dir, "v3_audit")


# --------------------------------------------------------------------------- #
# Figure 6: the price of gating
# --------------------------------------------------------------------------- #

CAUSES = (
    ("oom_at_cap", "GPU cap", BLUE),
    ("timeout", "timeout", ORANGE),
    ("environment", "environment", AQUA),
    ("agent_code", "agent code", YELLOW),
)


def price_figure_v3(
    waves_path: Path = HERE / "waves23.csv",
    crashes_path: Path = HERE / "crashes.csv",
    output_dir: Path = OUT,
) -> RenderedFigure:
    rows = _rows(waves_path)
    crash = {r["arm"]: r for r in _rows(crashes_path)}
    levels = sorted({r["level"] for r in rows}, key=_level_key)
    by = {lv: [r for r in rows if r["level"] == lv] for lv in levels}
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2))
    fig.patch.set_facecolor(SURFACE)
    half = 0.27

    def per_run(ax, values, title, unit, fmt, top):
        _axis(ax)
        ax.set_xlim(-0.6, len(levels) - 0.4)
        ax.set_ylim(0, top)
        means = [sum(v) / len(v) for v in values]
        for x, (lv, mean, seeds) in enumerate(zip(levels, means, values, strict=True)):
            _rounded_bar(ax, x - half, x + half, 0, mean, LEVEL_COLOR[lv])
            ax.scatter([x - 0.1, x + 0.1][: len(seeds)], seeds, s=14, color=INK,
                       edgecolors=SURFACE, linewidths=1.0, zorder=3)
            ax.text(x, max(max(seeds), mean) + top * 0.03, fmt(mean), ha="center",
                    va="bottom", fontsize=FONT, color=INK)
        ax.set_xticks(range(len(levels)), levels)
        ax.set_ylabel(unit, fontsize=FONT, color=INK2)
        _title(ax, f"{title}, L3/L0 {means[-1] / means[0]:.2f}\u00d7")

    per_run(axes[0], [[float(r["cost_usd"]) for r in by[lv]] for lv in levels],
            "(a) Cost/run", "US dollars", lambda v: f"{v:.2f}", 2.3)
    per_run(axes[1],
            [[(int(r["tokens_prompt"]) + int(r["tokens_completion"])) / 1e6 for r in by[lv]]
             for lv in levels],
            "(b) Tokens/run", "millions", lambda v: f"{v:.1f}", 10.5)

    ax = axes[2]
    _axis(ax)
    ax.set_xlim(-0.6, len(levels) - 0.4)
    ax.set_ylim(0, 62)
    width = 0.24
    for gate in (1, 2, 3):
        for x, lv in enumerate(levels):
            if gate > _level_key(lv):
                continue
            attempts = sum(int(r[f"g{gate}_attempts"]) for r in by[lv])
            fails = sum(int(r[f"g{gate}_fails"]) for r in by[lv])
            pos = x + (gate - 2) * (width + 0.03)
            if fails:
                _rounded_bar(ax, pos - width / 2, pos + width / 2, 0, fails, SURFACE,
                             hatch="////", hatch_color=GATE_COLOR[gate], rounded=False)
            if attempts > fails:
                _rounded_bar(ax, pos - width / 2, pos + width / 2, fails + 0.4, attempts,
                             GATE_COLOR[gate])
            ax.text(pos, attempts + 0.8, f"{fails}/{attempts}", ha="center", va="bottom",
                    fontsize=FONT, color=INK, rotation=90)
    ax.set_xticks(range(len(levels)), levels)
    ax.set_ylabel("attempts; label rejected/all", fontsize=FONT, color=INK2)
    _title(ax, "(c) Gate attempts")
    ax.legend(
        handles=[Rectangle((0, 0), 1, 1, facecolor=GATE_COLOR[g], label=f"Gate {g}")
                 for g in (1, 2, 3)]
        + [Rectangle((0, 0), 1, 1, facecolor=SURFACE, hatch="////", edgecolor=MUTED,
                     label="rejected")],
        loc="upper left", fontsize=FONT, frameon=False, ncol=2, handlelength=1.0,
        columnspacing=0.6, handletextpad=0.3, borderaxespad=0.0,
    )

    ax = axes[3]
    _axis(ax)
    ax.set_xlim(-0.6, len(levels) - 0.4)
    ax.set_ylim(0, 27)
    for x, lv in enumerate(levels):
        bottom = 0.0
        present = [(k, c) for k, _, c in CAUSES if int(crash[lv][k])]
        for i, (key, color) in enumerate(present):
            count = int(crash[lv][key])
            top = bottom + count
            _rounded_bar(ax, x - half, x + half, bottom + (0.25 if i else 0), top, color,
                         rounded=i == len(present) - 1)
            bottom = top
        ax.text(x, bottom + 0.5, f"{crash[lv]['crashed']}/{crash[lv]['executions']}",
                ha="center", va="bottom", fontsize=FONT, color=INK)
    ax.set_xticks(range(len(levels)), levels)
    ax.set_ylabel("crashed; label crashed/all", fontsize=FONT, color=INK2)
    _title(ax, "(d) Crashes")
    ax.legend(handles=[Rectangle((0, 0), 1, 1, facecolor=c, label=n) for _, n, c in CAUSES],
              loc="upper right", fontsize=FONT, frameon=False, ncol=2, handlelength=1.0,
              columnspacing=0.6, handletextpad=0.3, borderaxespad=0.0)
    fig.tight_layout(pad=0.4, w_pad=0.9)
    return _save(fig, output_dir, "v3_price")



# --------------------------------------------------------------------------- #
# Figure 7: every run over time (run-based, from paper/timeline.py)
# --------------------------------------------------------------------------- #

TIMELINE = HERE.parent / ".cache" / "paper" / "timeline.csv"
STRIP_EVENTS = ("exec", "gate1", "gate2", "gate3")


def timeline_figure_v3(timeline_path: Path = TIMELINE,
                       output_dir: Path = OUT) -> RenderedFigure | None:
    """Cumulative spend of each run against its own clock, and every attempt on a strip.

    Skipped when the timeline has not been extracted on this machine: it is wave
    data, built by paper/timeline.py into .cache/, and never committed.
    """
    if not timeline_path.exists():
        return None
    rows = _rows(timeline_path)
    seeds = sorted({r["seed"] for r in rows})
    levels = sorted({r["level"] for r in rows}, key=_level_key)
    fig = plt.figure(figsize=(7.0, 2.8))
    fig.patch.set_facecolor(SURFACE)
    grid = fig.add_gridspec(2, len(seeds), height_ratios=[2.0, 1.05], hspace=0.12, wspace=0.12)
    t_max = max(float(r["t_h"]) for r in rows) * 1.16
    c_max = max(float(r["cost_usd"]) for r in rows) * 1.18
    for col, seed in enumerate(seeds):
        line_ax = fig.add_subplot(grid[0, col])
        strip_ax = fig.add_subplot(grid[1, col], sharex=line_ax)
        _axis(line_ax)
        line_ax.set_xlim(0, t_max)
        line_ax.set_ylim(0, c_max)
        for lv in levels:
            run = [r for r in rows if r["seed"] == seed and r["level"] == lv]
            ts = [0.0] + [float(r["t_h"]) for r in run]
            cs = [0.0] + [float(r["cost_usd"]) for r in run]
            line_ax.plot(ts, cs, color=LEVEL_COLOR[lv], linewidth=2,
                         solid_joinstyle="round", solid_capstyle="round")
            line_ax.scatter([ts[-1]], [cs[-1]], s=22, color=LEVEL_COLOR[lv], zorder=3,
                            edgecolors=SURFACE, linewidths=1.2)
            line_ax.annotate(f"{lv} ${cs[-1]:.2f}", (ts[-1], cs[-1]), xytext=(4, 0),
                             textcoords="offset points", va="center", fontsize=FONT,
                             color=INK, clip_on=False)
        line_ax.tick_params(axis="x", labelbottom=False)
        if col == 0:
            line_ax.set_ylabel("cumulative cost, US$", fontsize=FONT, color=INK2)
        else:
            line_ax.tick_params(axis="y", labelleft=False)
        _title(line_ax, f"({'ab'[col]}) Seed {seed}: spend against the run's clock")

        _axis(strip_ax, "x")
        strip_ax.set_ylim(len(levels) - 0.5, -0.5)
        strip_ax.set_yticks(range(len(levels)), levels if col == 0 else [""] * len(levels))
        for y, lv in enumerate(levels):
            run = [r for r in rows if r["seed"] == seed and r["level"] == lv
                   and r["event"] in STRIP_EVENTS]
            end = max(float(r["t_h"]) for r in rows if r["seed"] == seed and r["level"] == lv)
            strip_ax.plot([0, end], [y, y], color=GRID, linewidth=0.8, zorder=1)
            for r in run:
                t = float(r["t_h"])
                if r["event"] == "exec":
                    color = CRITICAL if r["outcome"] == "crash" else MUTED
                    strip_ax.scatter([t], [y], marker="x", s=16, color=color, linewidths=1.2,
                                     zorder=3)
                    continue
                gate = int(r["event"][-1])
                offset = (gate - 2) * 0.22
                accepted = r["outcome"] == "pass"
                strip_ax.scatter([t], [y + offset], marker="o", s=16, zorder=3,
                                 facecolors=GATE_COLOR[gate] if accepted else SURFACE,
                                 edgecolors=GATE_COLOR[gate], linewidths=1.0)
        strip_ax.set_xlabel("hours since the run started", fontsize=FONT, color=INK2)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=4.5, markerfacecolor=GATE_COLOR[g],
                   markeredgecolor=GATE_COLOR[g], label=f"Gate {g} accepted") for g in (1, 2, 3)
    ] + [
        plt.Line2D([], [], marker="o", linestyle="", markersize=4.5, markerfacecolor=SURFACE,
                   markeredgecolor=INK2, label="rejected (hollow)"),
        plt.Line2D([], [], marker="x", linestyle="", markersize=4.5, color=CRITICAL,
                   label="ungated run crashed"),
        plt.Line2D([], [], marker="x", linestyle="", markersize=4.5, color=MUTED,
                   label="ungated run ended"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=FONT,
               handletextpad=0.2, columnspacing=0.9, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.07, right=0.93, top=0.92, bottom=0.23)
    return _save(fig, output_dir, "v3_timeline")

def render_all(output_dir: Path = OUT) -> list[RenderedFigure]:
    return [
        key_results_figure(output_dir=output_dir),
        mechanism_figure_v3(output_dir=output_dir),
        redteam_figure_v3(output_dir=output_dir),
        audit_figure_v3(output_dir=output_dir),
        price_figure_v3(output_dir=output_dir),
    ] + [f for f in (timeline_figure_v3(output_dir=output_dir),) if f is not None]


if __name__ == "__main__":
    done = render_all()
    print(f"wrote {len(done)} figures in {OUT}: " + ", ".join(f.png.stem for f in done))
