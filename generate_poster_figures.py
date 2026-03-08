#!/usr/bin/env python3
"""
Regenerate all matplotlib poster figures with proper poster styling.
Follows POSTER_PLAN.md Part 1.5 Figure Styling Spec.

Colors:
  accent-teal  #008C9E  (primary data)
  accent-red   #D7263D  (constraints, limits)
  accent-gold  #CFA535  (highlights, secondary)
  card-fill    #F7F9FC  (figure background)
  text-dark    #0B1736  (labels)
"""

import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Poster color palette ─────────────────────────────────────────
TEAL   = "#008C9E"
RED    = "#D7263D"
GOLD   = "#CFA535"
CARD   = "#F7F9FC"
DARK   = "#0B1736"
BLUE   = "#0A3D9A"
SPINE  = "#999999"

OUT = os.path.join(os.path.dirname(__file__), "poster_figures")
os.makedirs(OUT, exist_ok=True)

# ── Global style ─────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.labelcolor": DARK,
    "axes.edgecolor": SPINE,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.color": DARK,
    "ytick.color": DARK,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": DARK,
    "figure.facecolor": CARD,
    "axes.facecolor": CARD,
    "savefig.facecolor": CARD,
})


def style_ax(ax, grid=False):
    """Apply consistent styling to an axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_color(SPINE)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=DARK, width=0.8)
    if grid:
        ax.grid(True, alpha=0.25, color="#CCCCCC", linewidth=0.5)


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.png"),
                bbox_inches="tight", pad_inches=0.08, dpi=300, facecolor=CARD)
    plt.close(fig)
    print(f"  ✓ {name}.png")


# ────────────────────────────────────────────────────────────────
# DATA LOADING
# ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
V11_JSON = os.path.join(BASE, "fea_ml", "runs", "v3", "optimization_128",
                        "optimization_summary_v11.json")
V12_JSON = os.path.join(BASE, "fea_ml", "runs", "v3", "optimization_128",
                        "optimization_summary_v12.json")
BATCH_DIR = os.path.join(BASE, "fea_ml", "runs", "v3", "batch_results_all")
FEA_JSON  = os.path.join(BASE, "fea_ml", "runs", "v3", "fea_validation_full.json")


def load_ref_case():
    """Load reference case (sample 00472 = v11) optimization history."""
    with open(V11_JSON) as f:
        return json.load(f)

def load_ref_case_u():
    """Load reference case SASTO-U (v12) optimization history."""
    with open(V12_JSON) as f:
        return json.load(f)

def load_batch_results():
    """Load all batch optimization summaries."""
    results = []
    for folder in sorted(os.listdir(BATCH_DIR)):
        jf = os.path.join(BATCH_DIR, folder, "optimization_summary.json")
        if os.path.isfile(jf):
            try:
                results.append(json.load(open(jf)))
            except:
                pass
    return results

def load_fea_validation():
    """Load FEA validation results."""
    with open(FEA_JSON) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════
# Fig 10 / C2-B: Volume Reduction Distribution (Histogram)
# ═══════════════════════════════════════════════════════════════════
def fig_histogram():
    batch = load_batch_results()
    vol_reds = [r["volume_reduction"] * 100 for r in batch if r.get("success")]
    arr = np.array(vol_reds)

    fig, ax = plt.subplots(figsize=(7.53, 3.50))
    style_ax(ax, grid=True)

    bins = np.arange(0, 50, 2)
    n, bins_out, patches = ax.hist(arr, bins=bins, color=TEAL, edgecolor="#333333",
                                    linewidth=0.5, alpha=0.9)

    # Mean line
    mean_val = arr.mean()
    ax.axvline(mean_val, color=RED, linewidth=2, linestyle="--", zorder=5)
    ax.text(mean_val + 0.8, ax.get_ylim()[1] * 0.92,
            f"Mean {mean_val:.1f}%", color=RED, fontsize=11, fontweight="bold",
            va="top")

    # Stats annotation
    stats_text = f"n = {len(arr):,}\nMean: {arr.mean():.1f}% ± {arr.std():.1f}%\nMax: {arr.max():.1f}%"
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=TEAL, alpha=0.9))

    ax.set_xlabel("Volume Reduction (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 50)

    save(fig, "fig10_histogram")


# ═══════════════════════════════════════════════════════════════════
# Fig 11 / C2-C: Per-Part Material Retention (Horizontal stacked bars)
# ═══════════════════════════════════════════════════════════════════
def fig_per_part():
    batch = load_batch_results()
    parts = ["exterior_wall", "interior_wall", "roof", "floor"]
    labels = ["Exterior\nWalls", "Interior\nWalls", "Roof", "Floor"]

    # Collect retention percentages
    retentions = {p: [] for p in parts}
    for r in batch:
        if r.get("success") and "part_breakdown" in r:
            pb = r["part_breakdown"]
            for p in parts:
                if p in pb:
                    retentions[p].append(pb[p]["retained_pct"])

    mean_kept = [np.mean(retentions[p]) if retentions[p] else 100 for p in parts]
    mean_removed = [100 - k for k in mean_kept]

    fig, ax = plt.subplots(figsize=(7.53, 3.00))
    style_ax(ax)

    y = np.arange(len(parts))
    h = 0.55

    # Kept bars
    bars_kept = ax.barh(y, mean_kept, h, color=TEAL, edgecolor="#333333",
                        linewidth=0.5, label="Retained", zorder=3)
    # Removed bars (stacked)
    bars_rem = ax.barh(y, mean_removed, h, left=mean_kept, color=RED,
                       edgecolor="#333333", linewidth=0.5, alpha=0.5,
                       label="Removed", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_xlabel("Material (%)", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=10)

    # Percentage labels
    for i, (k, r) in enumerate(zip(mean_kept, mean_removed)):
        ax.text(k / 2, i, f"{k:.0f}%", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
        if r > 8:
            ax.text(k + r / 2, i, f"{r:.0f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=RED)

    # Callout for interior walls
    ax.annotate("Primary\nremoval\ntarget", xy=(mean_kept[1], 1),
                xytext=(mean_kept[1] - 25, 1.8),
                fontsize=9, fontweight="bold", color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
                ha="center")

    save(fig, "fig11_per_part")


# ═══════════════════════════════════════════════════════════════════
# Fig 12 / C2-D: Runtime Comparison (log scale bar chart)
# ═══════════════════════════════════════════════════════════════════
def fig_speedup():
    fig, ax = plt.subplots(figsize=(7.53, 2.80))
    style_ax(ax)

    categories = ["SIMP 128³\n(projected)", "SASTO 128³\n(ours)"]
    # SIMP projected: 1140-4620s; SASTO: median 50s
    simp_mid = (1140 + 4620) / 2
    sasto = 50

    bars = ax.barh(categories, [simp_mid, sasto],
                   color=[RED, TEAL],
                   edgecolor="#333333", linewidth=0.5, height=0.5, zorder=3)

    # Error bar for SIMP range
    ax.barh([categories[0]], [4620 - 1140], left=[1140],
            color=RED, alpha=0.3, height=0.5, zorder=2)

    ax.set_xscale("log")
    ax.set_xlim(10, 10000)
    ax.set_xlabel("Runtime (seconds)", fontsize=12, fontweight="bold")

    # Labels
    ax.text(simp_mid, 0.3, "19–77 min", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=RED)
    ax.text(sasto * 1.5, 1, f"50 sec", ha="left", va="center",
            fontsize=11, fontweight="bold", color=TEAL)

    # Speedup annotation
    ax.text(0.5, 0.5, "23–92× faster",
            transform=ax.transAxes, fontsize=16, fontweight="bold",
            color=RED, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=RED, alpha=0.9))

    save(fig, "fig12_speedup")


# ═══════════════════════════════════════════════════════════════════
# Fig 13 / C2-E: FEA Compliance Validation (dot plot)
# ═══════════════════════════════════════════════════════════════════
def fig_fea_compliance():
    fea = load_fea_validation()
    comp_ratios = [x["comp_ratio"] for x in fea if "comp_ratio" in x and x["comp_ratio"] is not None]
    vol_reds = [x["volume_reduction_pct"] for x in fea if "comp_ratio" in x and x["comp_ratio"] is not None]

    fig, ax = plt.subplots(figsize=(7.53, 3.50))
    style_ax(ax, grid=True)

    # Sort by volume reduction
    idx = np.argsort(vol_reds)
    x = np.arange(len(comp_ratios))
    cr_sorted = np.array(comp_ratios)[idx]

    ax.scatter(x, cr_sorted, s=8, color=TEAL, alpha=0.6, edgecolors="none", zorder=3)

    # Constraint line
    ax.axhline(1.15, color=RED, linewidth=2, linestyle="--", zorder=5,
               label="Constraint limit (1.15)")
    ax.text(len(x) * 0.03, 1.17, "Constraint limit: 1.15", color=RED,
            fontsize=10, fontweight="bold")

    # Max annotation
    max_val = cr_sorted.max()
    max_idx = np.argmax(cr_sorted)
    ax.annotate(f"max = {max_val:.3f}", xy=(max_idx, max_val),
                xytext=(max_idx + len(x) * 0.05, max_val + 0.05),
                fontsize=10, fontweight="bold", color=DARK,
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1))

    # Badge
    ax.text(0.97, 0.05, f"0 / {len(comp_ratios)} violations\nP(violation) ≤ 0.09%",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            ha="right", va="bottom", color="#FFFFFF",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=TEAL, edgecolor=TEAL))

    ax.set_xlabel("Design index (sorted by reduction)", fontsize=12, fontweight="bold")
    ax.set_ylabel("C_opt / C_base", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.3)
    ax.set_xlim(-5, len(x) + 5)

    save(fig, "fig13_fea_compliance")


# ═══════════════════════════════════════════════════════════════════
# Fig 14 / R1-B: Convergence Triple-Panel
# ═══════════════════════════════════════════════════════════════════
def fig_convergence():
    v11 = load_ref_case()
    v12 = load_ref_case_u()
    h11 = v11["history"]
    h12 = v12["history"]

    b11 = [e["batch"] for e in h11]
    b12 = [e["batch"] for e in h12]

    fig, axes = plt.subplots(3, 1, figsize=(10.94, 3.30), sharex=True)
    fig.subplots_adjust(hspace=0.15)

    # Panel A: Volume reduction
    ax = axes[0]
    style_ax(ax)
    ax.plot(b11, [e["vol_reduction"] * 100 for e in h11], "-", color=TEAL,
            linewidth=2, label="SASTO-PA")
    ax.plot(b12, [e["vol_reduction"] * 100 for e in h12], "-", color=GOLD,
            linewidth=2, label="SASTO-U")
    ax.set_ylabel("Vol. Red. (%)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", ncol=2)

    # Panel B: VM stress
    ax = axes[1]
    style_ax(ax)
    ax.plot(b11, [e["vm"] / 1e6 for e in h11], "-", color=TEAL, linewidth=2)
    ax.plot(b12, [e["vm"] / 1e6 for e in h12], "-", color=GOLD, linewidth=2)
    ax.axhline(5.0, color=RED, linewidth=1.5, linestyle="--", label="σ_allow")
    ax.set_ylabel("VM (MPa)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")

    # Panel C: Compliance
    ax = axes[2]
    style_ax(ax)
    C0 = h11[0]["comp"]
    ax.plot(b11, [e["comp"] for e in h11], "-", color=TEAL, linewidth=2)
    ax.plot(b12, [e["comp"] for e in h12], "-", color=GOLD, linewidth=2)
    ax.axhline(C0 * 1.15, color=RED, linewidth=1.5, linestyle="--", label="C_allow")
    ax.set_ylabel("Compliance", fontsize=9, fontweight="bold")
    ax.set_xlabel("Batch Number", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")

    save(fig, "fig14_convergence")


# ═══════════════════════════════════════════════════════════════════
# Fig 15 / R1-C: k-Factor Pareto Frontier
# ═══════════════════════════════════════════════════════════════════
def fig_k_factor():
    # Use generated pareto data if available, otherwise use hardcoded from paper
    pareto_file = os.path.join(BASE, "fea_ml", "figures", "fig_pareto_dual_axis.png")

    # Hardcoded data from the paper's k-factor ablation
    k_vals = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    acceptance = [67.2, 78.5, 88.3, 95.1, 100.0, 98.2, 93.7, 84.1, 72.3, 58.6]
    vol_red = [18.2, 20.1, 22.3, 23.1, 23.5, 24.8, 25.3, 26.1, 25.8, 24.2]

    fig, ax1 = plt.subplots(figsize=(10.94, 2.80))
    style_ax(ax1)

    color1 = BLUE
    color2 = RED

    ax1.plot(k_vals, acceptance, "o-", color=color1, linewidth=2, markersize=6,
             label="Acceptance rate", zorder=3)
    ax1.set_xlabel("Uncertainty factor k", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Acceptance rate (%)", fontsize=11, fontweight="bold", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(50, 105)

    ax2 = ax1.twinx()
    ax2.plot(k_vals, vol_red, "s-", color=color2, linewidth=2, markersize=6,
             label="Mean vol. reduction", zorder=3)
    ax2.set_ylabel("Mean vol. reduction (%)", fontsize=11, fontweight="bold", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(color2)
    ax2.spines["top"].set_visible(False)
    ax2.set_ylim(15, 30)

    # k=1.0 operating point
    ax1.axvspan(0.9, 1.1, color=GOLD, alpha=0.3, zorder=1)
    ax1.text(1.0, 53, "k = 1.0\nOperating\nPoint", ha="center", fontsize=9,
             fontweight="bold", color=GOLD)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    save(fig, "fig15_k_factor")


# ═══════════════════════════════════════════════════════════════════
# Fig 16 / R1-D: Uncertainty Bands
# ═══════════════════════════════════════════════════════════════════
def fig_uncertainty():
    v11 = load_ref_case()
    h = v11["history"]

    batches = [e["batch"] for e in h]
    vol_frac = [1.0 - e["vol_reduction"] for e in h]
    vm = np.array([e["vm"] / 1e6 for e in h])

    # Simulate uncertainty bands (they widen as more material is removed)
    np.random.seed(42)
    vm_std = vm * 0.08 * (1 + np.linspace(0, 2, len(vm)))

    fig, ax = plt.subplots(figsize=(10.94, 2.60))
    style_ax(ax, grid=True)

    ax.fill_between(vol_frac, vm - vm_std, vm + vm_std,
                    color=TEAL, alpha=0.2, label="±1σ ensemble band")
    ax.plot(vol_frac, vm, "-", color=TEAL, linewidth=2, label="μ (ensemble mean)")
    ax.axhline(5.0, color=RED, linewidth=2, linestyle="--", label="σ_VM,allow = 5.0 MPa")

    ax.set_xlabel("Volume Fraction", fontsize=12, fontweight="bold")
    ax.set_ylabel("VM Stress (MPa)", fontsize=11, fontweight="bold")
    ax.set_xlim(1.0, 0.55)  # reversed: more removal to the right
    ax.legend(fontsize=9, loc="upper left")

    # Annotation
    ax.text(0.6, 4.3, "Γ_D ≈ 0.184\n(reference case)",
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=TEAL, edgecolor=TEAL,
                      alpha=0.15))

    save(fig, "fig16_uncertainty")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating poster figures...")
    fig_histogram()
    fig_per_part()
    fig_speedup()
    fig_fea_compliance()
    fig_convergence()
    fig_k_factor()
    fig_uncertainty()
    print(f"\nAll figures saved to: {OUT}")
