"""All plot figures, drawn to the conventions in v2/g4/FIGURE_CONVENTIONS.md.

Body:
  coverage.pdf     coverage against depth, failure region shaded
  shelf-life.pdf   transfer matrix + shelf life
  auc.pdf          mu beats sigma, below-chance region shaded

Appendix:
  per-target.pdf   the three targets separately
  width.pdf        what conditioning costs in interval width
  mechanism.pdf    bias inversion and the sigma/error growth gap

Conventions: (a)/(b) below panels, legends inline as patch swatches, shaded
regions carry meaning, value labels only where the number is the point.
Everything reads frozen adjudication records; nothing is recomputed here.
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
TARGETS = ["compliance", "max_displacement", "max_von_mises"]
TARGET_LABEL = {"compliance": "Compliance",
                "max_displacement": "Max displacement",
                "max_von_mises": "Max von Mises"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 7.6, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "figure.dpi": 200,
})
INK, ACCENT, MUTED = "#1a1a1a", "#b02418", "#41668c"
FAIL_BG = "#f2d9d5"
BAND = "#c9d6e4"
TCOLOR = {"compliance": "#b02418", "max_displacement": "#41668c",
          "max_von_mises": "#7a8b5a"}


def wilson(x, n, two_sided=True):
    z = 1.959963984540054 if two_sided else 1.6448536269514722
    p = x / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - r) / d), float((c + r) / d)


def tidy(ax, ylab: str, xlab: str = "Material removed (percent)") -> None:
    ax.set_xticks(range(len(BINS)))
    ax.set_xticklabels(SHORT)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.4, alpha=0.30)
    ax.set_axisbelow(True)


def panel_label(ax, text: str) -> None:
    ax.text(0.5, -0.30, text, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.5)


# ----------------------------------------------------------------- body


def fig_coverage(rep) -> None:
    per = {r["bin_label"]: r for r in rep["adjudication"]["per_bin"]}
    cov = np.array([per[b]["coverage"] for b in BINS])
    n = np.array([per[b]["n"] for b in BINS])
    cv = np.array([per[b]["covered"] for b in BINS])
    lo, hi = zip(*[wilson(c, m) for c, m in zip(cv, n)])
    x = np.arange(len(BINS))

    fig, ax = plt.subplots(figsize=(4.7, 2.75))
    ax.axhspan(0.55, 0.95, color=FAIL_BG, zorder=0)
    ax.axhline(0.95, color=INK, ls="--", lw=0.9, zorder=2)
    ax.text(0.06, 0.9515, "Target coverage 0.95", fontsize=7.4, va="bottom",
            transform=ax.get_yaxis_transform())
    ax.text(0.06, 0.573, "Below target", fontsize=7.4, color="#8c4038",
            va="bottom", transform=ax.get_yaxis_transform())

    ax.fill_between(x, lo, hi, color=BAND, zorder=2, lw=0)
    ax.plot(x, cov, "o-", color=ACCENT, lw=1.7, markersize=4.6, zorder=4)
    ax.annotate(f"{cov[-1]:.3f}", (x[-1], cov[-1]), textcoords="offset points",
                xytext=(-10, -14), ha="right", fontsize=8, color=ACCENT)

    tidy(ax, "Joint coverage")
    ax.set_ylim(0.55, 1.02)
    ax.set_xlim(-0.35, len(BINS) - 0.65)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGS / "coverage.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote coverage.pdf")


def fig_shelf(shelf) -> None:
    """Transfer matrix only.

    The bar panel that used to sit beside this was a second reading of the
    matrix's own upper triangle, so it was removed.
    """
    m = shelf["experiment_a"]["matrix"]
    life = shelf["experiment_a"]["shelf_life_bins"]
    grid = np.array([[m[bi][bj]["coverage"] for bj in BINS] for bi in BINS])
    lower = np.array([[m[bi][bj]["wilson_lower"] for bj in BINS] for bi in BINS])

    fig, ax = plt.subplots(figsize=(4.3, 3.1))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0.60, vmax=1.0, aspect="auto")
    for i in range(len(BINS)):
        for j in range(len(BINS)):
            valid = lower[i, j] >= 0.95
            ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                    fontsize=7.0, color="black",
                    fontweight="bold" if valid else "normal")
            if valid:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="black", lw=1.3))
    # shelf life annotated on the matrix itself
    for i, b in enumerate(BINS[:-1]):
        ax.text(len(BINS) - 0.35, i, f"{life[b] * 5}", ha="left", va="center",
                fontsize=7.2, color=MUTED)
    ax.text(len(BINS) - 0.35, -0.72, "Shelf\nlife", ha="left", va="center",
            fontsize=7.0, color=MUTED, linespacing=1.15)

    ax.set_xticks(range(len(BINS))); ax.set_xticklabels(SHORT)
    ax.set_yticks(range(len(BINS))); ax.set_yticklabels(SHORT)
    ax.set_xlabel("Evaluated at (percent removed)")
    ax.set_ylabel("Calibrated at (percent removed)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.10)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.6)
    cb.set_label("Coverage", fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGS / "shelf-life.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote shelf-life.pdf")


def fig_auc(shelf) -> None:
    per = shelf["experiment_b"]["per_bin"]
    sig = [per[b]["sigma_mean_all"] for b in BINS]
    mu = [per[b]["mu_compliance"] for b in BINS]
    x = np.arange(len(BINS))

    fig, ax = plt.subplots(figsize=(4.7, 2.75))
    ax.axhspan(0.28, 0.5, color=FAIL_BG, zorder=0)
    ax.axhline(0.5, color=INK, ls="--", lw=0.9, zorder=2)
    ax.text(0.05, 0.507, "Chance", fontsize=7.4, va="bottom",
            transform=ax.get_yaxis_transform())
    ax.text(0.05, 0.293, "Worse than chance", fontsize=7.4, color="#8c4038",
            va="bottom", transform=ax.get_yaxis_transform())

    ax.plot(x, mu, "o-", color=ACCENT, lw=1.7, markersize=4.6, zorder=4,
            label=r"Predicted mean $\mu$")
    ax.plot(x, sig, "s--", color=MUTED, lw=1.7, markersize=4.4, zorder=4,
            label=r"Ensemble spread $\sigma$")
    ax.annotate(f"{mu[-1]:.3f}", (x[-1], mu[-1]), textcoords="offset points",
                xytext=(-7, 7), ha="right", fontsize=8, color=ACCENT)
    ax.annotate(f"{sig[-1]:.3f}", (x[-1], sig[-1]), textcoords="offset points",
                xytext=(-10, -13), ha="right", fontsize=8, color=MUTED)

    tidy(ax, "AUC for predicting failure")
    ax.set_ylim(0.28, 0.80)
    ax.set_xlim(-0.35, len(BINS) - 0.65)
    ax.legend(frameon=False, loc="upper left", handlelength=1.7,
              borderaxespad=0.3)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGS / "auc.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote auc.pdf")


# ------------------------------------------------------------- appendix


def fig_per_target(rep) -> None:
    per = {r["bin_label"]: r for r in rep["adjudication"]["per_bin"]}
    fig, ax = plt.subplots(figsize=(4.9, 2.8))
    x = np.arange(len(BINS))
    ax.axhspan(0.45, 0.95, color=FAIL_BG, zorder=0)
    ax.axhline(0.95, color=INK, ls="--", lw=0.9, zorder=2)
    ax.text(0.05, 0.9525, "Target 0.95", fontsize=7.4, va="bottom",
            transform=ax.get_yaxis_transform())
    for t in TARGETS:
        vals = [per[b]["per_target_covered"][t] / per[b]["n"] for b in BINS]
        ax.plot(x, vals, "o-", color=TCOLOR[t], lw=1.6, markersize=4.2,
                zorder=4, label=TARGET_LABEL[t])
    tidy(ax, "Per-target coverage")
    ax.set_ylim(0.45, 1.02)
    ax.set_xlim(-0.35, len(BINS) - 0.55)
    ax.legend(frameon=False, loc="lower left", handlelength=1.7,
              borderaxespad=0.3)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGS / "per-target.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote per-target.pdf")


def fig_width(arms) -> None:
    """What depth-conditional calibration costs, per target and per depth.

    Absolute quantiles, not ratios: the shallowest von Mises quantile is
    negative, so a ratio against it would be meaningless.
    """
    q = arms["arm_b"]["per_bin_q"]
    fig, ax = plt.subplots(figsize=(4.9, 2.8))
    x = np.arange(len(BINS))
    ax.axhline(0.0, color=INK, lw=0.8, alpha=0.6)
    for t in TARGETS:
        vals = [q[b][t] for b in BINS]
        ax.plot(x, vals, "o-", color=TCOLOR[t], lw=1.6, markersize=4.2,
                label=TARGET_LABEL[t])
    tidy(ax, "Depth-conditional $q$ (normalized log)")
    ax.set_xlim(-0.35, len(BINS) - 0.55)
    ax.legend(frameon=False, loc="upper left", handlelength=1.7,
              borderaxespad=0.3)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGS / "width.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote width.pdf")


def fig_spatial(sp) -> None:
    """Where the optimizer removes material. Descriptive, amendment 06."""
    h = [x for x in sp["removal_by_height"] if x is not None]
    pr = [x for x in sp["removal_by_perimeter_distance"] if x is not None]
    nb = sp["removal_by_neighbour_count"]
    cnt = sp["voxels_by_neighbour_count"]

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.35))

    ax = axes[0]
    y = np.linspace(0, 1, len(h))
    ax.plot(h, y, "o-", color=ACCENT, lw=1.6, markersize=3.8)
    ax.set_xlabel("Fraction removed")
    ax.set_ylabel("Normalized height")
    ax.set_xlim(0, max(h) * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(lw=0.4, alpha=0.30); ax.set_axisbelow(True)
    panel_label(ax, "(a)")

    ax = axes[1]
    xs = np.arange(len(pr))
    ax.plot(xs, pr, "o-", color=MUTED, lw=1.6, markersize=3.8)
    ax.set_xlabel("Voxels from footprint edge")
    ax.set_ylabel("Fraction removed")
    ax.set_xticks(xs)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.4, alpha=0.30); ax.set_axisbelow(True)
    panel_label(ax, "(b)")

    ax = axes[2]
    ks = [k for k in range(7) if cnt[k] > 0 and nb[k] is not None]
    ax.bar(ks, [nb[k] for k in ks], color="#7a8b5a", width=0.66)
    ax.set_xlabel("Baseline 6-neighbour count")
    ax.set_ylabel("Fraction removed")
    ax.set_xticks(ks)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.4, alpha=0.30); ax.set_axisbelow(True)
    panel_label(ax, "(c)")

    fig.tight_layout(pad=0.4, w_pad=1.7)
    fig.savefig(FIGS / "spatial.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote spatial.pdf")


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    rep = json.loads((CONTROL / "k6-coverage-gb200-2096.json").read_text())
    shelf = json.loads((CONTROL / "k6-amendment-05-shelf-life.json").read_text())
    arms = json.loads((CONTROL / "k6-amendment-03-arms.json").read_text())

    fig_coverage(rep)
    fig_shelf(shelf)
    fig_auc(shelf)
    fig_per_target(rep)
    fig_width(arms)

    sp_path = CONTROL / "k6-spatial-pattern.json"
    if sp_path.exists():
        fig_spatial(json.loads(sp_path.read_text()))
    else:
        print("skipped spatial.pdf (amendment 06 record not present yet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
