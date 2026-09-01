"""Figure 1: the measurement protocol.

An explanatory schematic in the sense the peer papers use one (Wang and Ning,
NeurIPS 2025, Fig. 1; Stanton et al., AISTATS 2023, Fig. 2): it illustrates the
METHOD. It deliberately draws no geometry. An earlier version rendered cartoon
house glyphs with randomly placed red squares standing in for removed material,
which invented data beside a figure showing real hash-verified occupancy, and
contradicted the measured erosion pattern. Real geometry appears in Figure 2 and
the measured spatial pattern in the appendix.

Card A  the pipeline, fit to verify
Card B  the depth axis: where the calibration population sits, where the bound
        is queried, and how far a calibration carries

All numbers are read from the frozen records.
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

CARD_A = "#e4edf6"
CARD_B = "#fbeadf"
INK = "#1a1a1a"
STEEL = "#41668c"
ACCENT = "#b02418"

FS_TASK = 8.5
FS_HEAD = 8.0
FS_META = 7.5
FS_KEY = 7.0

M = 0.014
FW, FH = 7.2, 2.95

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


def stage_box(ax, x, y, w, h, title, lines, accent=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.02",
        facecolor="white", edgecolor=STEEL if accent else "#c3ced9",
        linewidth=1.1 if accent else 0.8, zorder=2))
    ax.text(x + w / 2, y + h - 0.10, title, ha="center", va="top",
            fontsize=FS_HEAD, fontweight="bold", zorder=3)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.28 - i * 0.155, ln, ha="center",
                va="top", fontsize=FS_KEY, color="#40505f", zorder=3)


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    shelf = json.loads((CONTROL / "k6-amendment-05-shelf-life.json").read_text())
    span = shelf["experiment_a"]["shelf_life_bins"]["(5,10%]"] * 5

    fig = plt.figure(figsize=(FW, FH))
    fig.patch.set_facecolor("white")

    RAIL = M + 0.020
    LEFT = RAIL + 0.022
    RIGHT = 1.0 - M - 0.012

    # ---------------- Card A: the pipeline ---------------------------
    ay1, ay0 = 1.0 - M, 0.505
    card(fig, M, ay0, 1.0 - M, ay1, CARD_A)
    fig.text(RAIL, (ay0 + ay1) / 2, "Protocol", rotation=90, ha="center",
             va="center", fontsize=FS_TASK, fontweight="bold")

    ax = fig.add_axes((LEFT, ay0 + 0.030, RIGHT - LEFT, ay1 - ay0 - 0.065))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_axis_off(); ax.patch.set_alpha(0)

    bw, bh, by = 0.205, 0.70, 0.24
    gap = (1.0 - 4 * bw) / 3
    xs = [i * (bw + gap) for i in range(4)]
    stage_box(ax, xs[0], by, bw, bh, "Fit",
              ["6,643 baselines", "5-member ensemble", "all at 0% removed"])
    stage_box(ax, xs[1], by, bw, bh, "Calibrate",
              ["1,108 baselines", r"$q$ at $\alpha/J$", "frozen by value"])
    stage_box(ax, xs[2], by, bw, bh, "Erode",
              ["hash-derived path", "no solver calls", "one state per band"])
    stage_box(ax, xs[3], by, bw, bh, "Verify",
              ["FEA on selected", "10,305 states", "coverage per band"],
              accent=True)
    for i in range(3):
        ax.add_patch(FancyArrowPatch(
            (xs[i] + bw + 0.008, by + bh / 2), (xs[i + 1] - 0.008, by + bh / 2),
            arrowstyle="-|>", mutation_scale=9, color="#7b8b9a", lw=1.3,
            shrinkA=0, shrinkB=0, zorder=4))
    ax.text(0.5, 0.075, r"bound  $U_j=\mu_j+\kappa_j\sigma_j+q_j$,   "
                        "fitted once and never refreshed",
            ha="center", va="center", fontsize=FS_META, color=INK)

    # ---------------- Card B: the depth axis -------------------------
    by1, by0 = ay0 - 0.020, M
    card(fig, M, by0, 1.0 - M, by1, CARD_B)
    fig.text(RAIL, (by0 + by1) / 2, "Depth", rotation=90, ha="center",
             va="center", fontsize=FS_TASK, fontweight="bold")

    ax = fig.add_axes((LEFT, by0 + 0.035, RIGHT - LEFT, by1 - by0 - 0.075))
    ax.set_xlim(-3.5, 40); ax.set_ylim(0, 1)
    ax.set_axis_off(); ax.patch.set_alpha(0)

    AXY = 0.40
    ax.annotate("", xy=(37.5, AXY), xytext=(-2.5, AXY),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))
    for t in (0, 10, 20, 30):
        ax.plot([t, t], [AXY - 0.055, AXY + 0.055], color=INK, lw=0.9)
        ax.text(t, AXY - 0.145, f"{t}%", ha="center", va="top", fontsize=FS_KEY)
    ax.text(39.5, AXY - 0.145, "Material removed", ha="right", va="top",
            fontsize=FS_META, color=INK)

    # the calibration population: a single point at zero
    ax.plot([0], [AXY], marker="o", markersize=7, color=STEEL, zorder=5)
    ax.annotate("Fit and calibration data\n6,643 + 1,108, all at 0%",
                xy=(0, AXY + 0.06), xytext=(0.5, AXY + 0.50),
                ha="left", va="bottom", fontsize=FS_KEY, color=STEEL,
                linespacing=1.3,
                arrowprops=dict(arrowstyle="-", color=STEEL, lw=0.8))

    # the queried population: a span
    ax.add_patch(plt.Rectangle((5, AXY + 0.035), 30, 0.075,
                               facecolor=ACCENT, alpha=0.28, edgecolor="none",
                               zorder=3))
    ax.annotate("Bound queried here\n10,305 verified states, 5-35%",
                xy=(26, AXY + 0.11), xytext=(20.5, AXY + 0.50),
                ha="left", va="bottom", fontsize=FS_KEY, color=ACCENT,
                linespacing=1.3,
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.8))

    # shelf life, read from the frozen record
    y = AXY - 0.30
    ax.plot([7.5, 7.5 + span], [y, y], color=INK, lw=1.1)
    for xx in (7.5, 7.5 + span):
        ax.plot([xx, xx], [y - 0.05, y + 0.05], color=INK, lw=1.1)
    ax.text(7.5 + span / 2, y - 0.085,
            f"Calibration at 5-10% stays valid for {span} more points",
            ha="center", va="top", fontsize=FS_KEY)

    fig.savefig(FIGS / "protocol.pdf", facecolor="white")
    plt.close(fig)
    print("wrote protocol.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
