"""Figures for the rebuilt manuscript.

Fig 1  coverage vs depth               the setup: where the bound breaks
Fig 2  transfer matrix + shelf life    the lead result
Fig 3  mu beats sigma (within-bin AUC) the second result

All read frozen adjudication records; nothing is recomputed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path("/Users/eric/workspace/sasto-modernization-control/v2/g4/paper")
FIGS = PAPER / "figures"
CONTROL = Path("/Users/eric/workspace/sasto-modernization-control/v2/g3")

SHORT = ["5-10", "10-15", "15-20", "20-25", ">25"]
BINS = ["(5,10%]", "(10,15%]", "(15,20%]", "(20,25%]", ">25%"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "figure.dpi": 200,
})
INK, ACCENT, MUTED = "#1a1a1a", "#b02418", "#5a7fa6"


def wilson(x, n, two_sided=True):
    z = 1.959963984540054 if two_sided else 1.6448536269514722
    p = x / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - r) / d), float((c + r) / d)


def fig_coverage(rep):
    per = {r["bin_label"]: r for r in rep["adjudication"]["per_bin"]}
    cov = [per[b]["coverage"] for b in BINS]
    n = [per[b]["n"] for b in BINS]
    cv = [per[b]["covered"] for b in BINS]
    lo, hi = zip(*[wilson(c, m) for c, m in zip(cv, n)])

    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    x = np.arange(len(BINS))
    ax.axhline(0.95, color=INK, ls="--", lw=0.9, zorder=1)
    ax.text(len(BINS) - 0.55, 0.958, "target 0.95", ha="right", va="bottom", fontsize=7.5)
    ax.errorbar(x, cov, yerr=[np.array(cov) - np.array(lo), np.array(hi) - np.array(cov)],
                fmt="o-", color=ACCENT, ecolor=ACCENT, elinewidth=1.1, capsize=3,
                markersize=5, lw=1.6, zorder=3)
    for xi, (c, m) in enumerate(zip(cov, n)):
        if c > 0.9:
            ax.annotate(f"{c:.3f}", (xi, c), textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=7.5, color=ACCENT)
        else:
            ax.annotate(f"{c:.3f}", (xi, c), textcoords="offset points", xytext=(-13, 6),
                        ha="right", fontsize=7.5, color=ACCENT)
        ax.annotate(f"n={m}", (xi, 0.596), ha="center", fontsize=7, color="#666666")
    ax.set_xticks(x); ax.set_xticklabels(SHORT)
    ax.set_xlabel("material removed (percent)")
    ax.set_ylabel("joint coverage")
    ax.set_ylim(0.575, 1.03); ax.set_xlim(-0.5, len(BINS) - 0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.4, alpha=0.35); ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGS / "coverage.pdf", bbox_inches="tight"); plt.close(fig)
    print("wrote coverage.pdf")


def fig_shelf(shelf):
    """Transfer matrix heatmap + shelf-life bars. The lead figure."""
    m = shelf["experiment_a"]["matrix"]
    life = shelf["experiment_a"]["shelf_life_bins"]
    grid = np.array([[m[bi][bj]["coverage"] for bj in BINS] for bi in BINS])
    lower = np.array([[m[bi][bj]["wilson_lower"] for bj in BINS] for bi in BINS])

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.75),
                             gridspec_kw={"width_ratios": [1.32, 1]})

    ax = axes[0]
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0.60, vmax=1.0, aspect="auto")
    for i in range(len(BINS)):
        for j in range(len(BINS)):
            valid = lower[i, j] >= 0.95
            ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    color="black", fontweight="bold" if valid else "normal")
            if valid:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="black", lw=1.4))
    ax.set_xticks(range(len(BINS))); ax.set_xticklabels(SHORT)
    ax.set_yticks(range(len(BINS))); ax.set_yticklabels(SHORT)
    ax.set_xlabel("evaluated at (percent removed)")
    ax.set_ylabel("calibrated at")
    ax.set_title("coverage transfer")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("coverage", fontsize=8)

    ax = axes[1]
    origins = BINS[:-1]
    vals = [life[b] * 5 for b in origins]
    y = np.arange(len(origins))
    ax.barh(y, vals, color=MUTED, height=0.6)
    for yi, v in enumerate(vals):
        ax.text(v + 0.4, yi, f"{v} pts", va="center", fontsize=8,
                color=INK if v else ACCENT)
    ax.set_yticks(y); ax.set_yticklabels(SHORT[:-1])
    ax.invert_yaxis()
    ax.set_xlabel("further removal before the bound fails (points)")
    ax.set_ylabel("calibrated at")
    ax.set_title("calibration shelf life")
    ax.set_xlim(0, 18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", lw=0.4, alpha=0.35); ax.set_axisbelow(True)

    fig.tight_layout(pad=0.4, w_pad=1.8)
    fig.savefig(FIGS / "shelf-life.pdf", bbox_inches="tight"); plt.close(fig)
    print("wrote shelf-life.pdf")


def fig_auc(shelf):
    per = shelf["experiment_b"]["per_bin"]
    sig = [per[b]["sigma_mean_all"] for b in BINS]
    mu = [per[b]["mu_compliance"] for b in BINS]
    x = np.arange(len(BINS))

    fig, ax = plt.subplots(figsize=(5.2, 2.75))
    ax.axhline(0.5, color=INK, ls="--", lw=0.9)
    ax.text(0.02, 0.505, "chance", fontsize=7.5, color=INK, transform=ax.get_yaxis_transform())
    ax.plot(x, mu, "o-", color=ACCENT, lw=1.7, markersize=5, label=r"predicted mean $\mu$")
    ax.plot(x, sig, "s-", color=MUTED, lw=1.7, markersize=5, label=r"ensemble spread $\sigma$")
    ax.annotate(f"{mu[-1]:.3f}", (x[-1], mu[-1]), textcoords="offset points",
                xytext=(-6, 9), ha="right", fontsize=8, color=ACCENT)
    ax.annotate(f"{sig[-1]:.3f}", (x[-1], sig[-1]), textcoords="offset points",
                xytext=(-10, 7), ha="right", fontsize=8, color=MUTED)
    ax.fill_between([3.5, 4.5], 0.30, 0.78, color="#cccccc", alpha=0.18, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(SHORT)
    ax.set_xlabel("material removed (percent)")
    ax.set_ylabel("within-bin AUC for predicting failure")
    ax.set_ylim(0.28, 0.80); ax.set_xlim(-0.35, len(BINS) - 0.65)
    ax.legend(frameon=False, loc="upper left", handlelength=1.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.4, alpha=0.35); ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGS / "auc.pdf", bbox_inches="tight"); plt.close(fig)
    print("wrote auc.pdf")


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    rep = json.loads((CONTROL / "k6-coverage-gb200-2096.json").read_text())
    shelf = json.loads((CONTROL / "k6-amendment-05-shelf-life.json").read_text())
    fig_coverage(rep)
    fig_shelf(shelf)
    fig_auc(shelf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
