"""Figure 2: the measurement protocol, drawn as a schematic.

This is an explanatory diagram, not data. It answers two questions the data
figures cannot: what the procedure actually is, and why the training support
sits in the wrong place relative to where the bound gets used.

Card A  the pipeline, left to right, from fit corpus to per-depth coverage.
Card B  the depth axis. Everything the model learned sits at 0% removed;
        everything it is asked about sits between 5% and 35%. The shelf-life
        bracket shows how far a calibration carries before it stops being
        valid, which is the paper's headline number.

Visual language matches the benchmark plate: same pastel cards, same palette,
same type and spacing scales, fixed canvas, no bbox_inches='tight'.
All numbers come from the frozen records, not from prose.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PAPER = Path("/Users/eric/workspace/sasto-modernization-control/v2/g4/paper")
FIGS = PAPER / "figures"
CONTROL = Path("/Users/eric/workspace/sasto-modernization-control/v2/g3")

# palette shared with the benchmark plate
CARD_A = "#e4edf6"
CARD_B = "#fbeadf"
INK = "#1a1a1a"
WALL = "#8fa3b8"
PART = "#cbbb98"
ROOF = "#8a6a55"
REMOVED = "#c0392b"
STEEL = "#41668c"

FS_TASK = 8.5
FS_HEAD = 8.0
FS_META = 7.5
FS_KEY = 7.0

M = 0.014
FW, FH = 7.2, 3.55

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8,
    "figure.dpi": 200,
})


def card(fig, x0, y0, x1, y1, color):
    bg = fig.add_axes((x0, y0, x1 - x0, y1 - y0), zorder=0)
    bg.set_axis_off()
    bg.add_patch(FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0, transform=bg.transAxes,
        boxstyle="round,pad=0,rounding_size=0.045",
        facecolor=color, edgecolor="none", clip_on=False))
    return bg


def house(ax, x, y, w, h, removed=0.0, seed=0):
    """Small schematic house glyph. Decorative, so it is drawn not replayed."""
    body_h = h * 0.58
    ax.add_patch(plt.Rectangle((x, y), w, body_h, facecolor=WALL,
                               edgecolor="none", zorder=2))
    ax.add_patch(plt.Polygon([[x - w * 0.08, y + body_h],
                              [x + w / 2, y + h],
                              [x + w * 1.08, y + body_h]],
                             facecolor=ROOF, edgecolor="none", zorder=2))
    if removed > 0:
        rng = np.random.default_rng(seed)
        for _ in range(int(removed * 26)):
            bx = x + rng.uniform(0.06, 0.88) * w
            by = y + rng.uniform(0.08, 0.82) * body_h
            ax.add_patch(plt.Rectangle((bx, by), w * 0.09, body_h * 0.13,
                                       facecolor=REMOVED, edgecolor="none",
                                       alpha=0.85, zorder=3))


def stage_box(ax, x, y, w, h, title, lines, accent=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.02",
        facecolor="white", edgecolor=STEEL if accent else "#c3ced9",
        linewidth=1.1 if accent else 0.8, zorder=2))
    ax.text(x + w / 2, y + h - 0.055, title, ha="center", va="top",
            fontsize=FS_HEAD, fontweight="bold", zorder=3)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.135 - i * 0.075, ln, ha="center",
                va="top", fontsize=FS_KEY, color="#40505f", zorder=3)


def arrow(ax, x0, y0, x1, y1, color="#7b8b9a", lw=1.3):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=9, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=4))


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    shelf = json.loads((CONTROL / "k6-amendment-05-shelf-life.json").read_text())
    life = shelf["experiment_a"]["shelf_life_bins"]

    fig = plt.figure(figsize=(FW, FH))
    fig.patch.set_facecolor("white")

    RAIL = M + 0.020
    LEFT = RAIL + 0.020

    # ---------------- Card A: the pipeline ---------------------------
    ay1, ay0 = 1.0 - M, 0.505
    card(fig, M, ay0, 1.0 - M, ay1, CARD_A)
    fig.text(RAIL, (ay0 + ay1) / 2, "Protocol", rotation=90, ha="center",
             va="center", fontsize=FS_TASK, fontweight="bold")

    ax = fig.add_axes((LEFT, ay0 + 0.02, 1.0 - M - 0.012 - LEFT, ay1 - ay0 - 0.05))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.patch.set_alpha(0)

    bw, bh, by = 0.185, 0.62, 0.20
    gap = (1.0 - 4 * bw) / 3
    xs = [i * (bw + gap) for i in range(4)]

    stage_box(ax, xs[0], by, bw, bh, "Fit",
              ["6,643 baselines", "5-member ensemble", "all at 0% removed"])
    stage_box(ax, xs[1], by, bw, bh, "Calibrate",
              ["1,108 baselines", r"$q$ at $\alpha/J$", "frozen by value"])
    stage_box(ax, xs[2], by, bw, bh, "Erode",
              ["hash-derived path", "no solver calls", "one state per band"])
    stage_box(ax, xs[3], by, bw, bh, "Verify", ["FEA on selected", "10,305 states",
                                                "coverage per band"], accent=True)
    for i in range(3):
        arrow(ax, xs[i] + bw + 0.008, by + bh / 2, xs[i + 1] - 0.008, by + bh / 2)

    ax.text(0.5, 0.055, r"bound  $U_j=\mu_j+\kappa_j\sigma_j+q_j$   "
                        "fitted once, never refreshed",
            ha="center", va="center", fontsize=FS_META, color=INK)

    # ---------------- Card B: the depth axis -------------------------
    by1, by0 = ay0 - 0.018, M
    card(fig, M, by0, 1.0 - M, by1, CARD_B)
    fig.text(RAIL, (by0 + by1) / 2, "Depth", rotation=90, ha="center",
             va="center", fontsize=FS_TASK, fontweight="bold")

    ax = fig.add_axes((LEFT, by0 + 0.03, 1.0 - M - 0.012 - LEFT, by1 - by0 - 0.06))
    ax.set_xlim(-3, 42); ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.patch.set_alpha(0)

    AXY = 0.34
    ax.annotate("", xy=(35.5, AXY), xytext=(-2, AXY),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))
    for t in (0, 10, 20, 30):
        ax.plot([t, t], [AXY - 0.035, AXY + 0.035], color=INK, lw=0.9)
        ax.text(t, AXY - 0.10, f"{t}%", ha="center", va="top", fontsize=FS_KEY)
    ax.text(41.5, AXY - 0.10, "material removed", ha="right", va="top",
            fontsize=FS_META, color=INK)

    # everything the model saw sits at zero
    house(ax, -1.6, AXY + 0.10, 3.2, 0.34, removed=0.0)
    ax.text(0, AXY + 0.52, "all training and\ncalibration data",
            ha="center", va="bottom", fontsize=FS_KEY, color=STEEL,
            linespacing=1.25)

    # where the bound is actually used
    for i, (d, r) in enumerate([(7.5, 0.10), (17.5, 0.35), (30, 0.75)]):
        house(ax, d - 1.6, AXY + 0.10, 3.2, 0.34, removed=r, seed=i + 1)
    ax.annotate("", xy=(34.5, AXY + 0.05), xytext=(5.0, AXY + 0.05),
                arrowprops=dict(arrowstyle="-|>", color=REMOVED, lw=1.2))
    ax.text(19.8, AXY + 0.60, "the bound is used here",
            ha="center", va="bottom", fontsize=FS_KEY, color=REMOVED)

    # shelf life, read off the frozen record
    span = life["(5,10%]"] * 5
    y = AXY - 0.20
    ax.plot([7.5, 7.5 + span], [y, y], color=INK, lw=1.1)
    for xx in (7.5, 7.5 + span):
        ax.plot([xx, xx], [y - 0.045, y + 0.045], color=INK, lw=1.1)
    ax.text(7.5 + span / 2, y - 0.075,
            f"shelf life at 5-10%: {span} more points",
            ha="center", va="top", fontsize=FS_KEY)

    fig.savefig(FIGS / "protocol.pdf", facecolor="white")
    plt.close(fig)
    print("wrote protocol.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
