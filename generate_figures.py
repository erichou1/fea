#!/usr/bin/env python3
"""
Generate all figures for the research paper.

Reads optimization history from JSON summaries and produces publication-quality
matplotlib figures saved as PNG and PDF.

Usage:
    python generate_figures.py

Output directory: figures/
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# ── Configuration ────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

V11_JSON = os.path.join(os.path.dirname(__file__),
                        "fea_ml", "runs", "v3", "optimization_128",
                        "optimization_summary_v11.json")  # SASTO-PA
V12_JSON = os.path.join(os.path.dirname(__file__),
                        "fea_ml", "runs", "v3", "optimization_128",
                        "optimization_summary_v12.json")  # SASTO-U

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_history(path):
    with open(path) as f:
        data = json.load(f)
    return data


# ═══════════════════════════════════════════════════════════════════
# Figure 4: Optimization Convergence (SASTO-PA vs SASTO-U)
# ═══════════════════════════════════════════════════════════════════
def fig4_convergence():
    v11 = load_history(V11_JSON)
    v12 = load_history(V12_JSON)

    h11 = v11["history"]
    h12 = v12["history"]

    batches_11 = [e["batch"] for e in h11]
    vol_red_11 = [e["vol_reduction"] * 100 for e in h11]
    vm_11 = [e["vm"] / 1e6 for e in h11]
    comp_11 = [e["comp"] for e in h11]

    batches_12 = [e["batch"] for e in h12]
    vol_red_12 = [e["vol_reduction"] * 100 for e in h12]
    vm_12 = [e["vm"] / 1e6 for e in h12]
    comp_12 = [e["comp"] for e in h12]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # Panel A: Volume reduction
    ax = axes[0]
    ax.plot(batches_11, vol_red_11, "-", color="#1565c0", linewidth=1.5,
            label="SASTO-PA (part-aware)")
    ax.plot(batches_12, vol_red_12, "--", color="#e65100", linewidth=1.5,
            label="SASTO-U (uniform)")
    ax.set_xlabel("Batch Number")
    ax.set_ylabel("Volume Reduction (%)")
    ax.set_title("(a) Volume Reduction")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 50)
    ax.axhline(y=45.0, color="#1565c0", alpha=0.3, linestyle=":")
    ax.axhline(y=34.3, color="#e65100", alpha=0.3, linestyle=":")

    # Panel B: Von Mises stress
    ax = axes[1]
    ax.plot(batches_11, vm_11, "-", color="#1565c0", linewidth=1.5,
            label="SASTO-PA (part-aware)")
    ax.plot(batches_12, vm_12, "--", color="#e65100", linewidth=1.5,
            label="SASTO-U (uniform)")
    ax.axhline(y=5.0, color="#c62828", linewidth=2, linestyle="-",
               alpha=0.7, label="$\\sigma_{VM,allow}$ = 5.0 MPa")
    ax.set_xlabel("Batch Number")
    ax.set_ylabel("Conservative VM Stress (MPa)")
    ax.set_title("(b) Von Mises Stress")
    ax.legend(loc="upper left", fontsize=9)

    # Panel C: Compliance
    ax = axes[2]
    C0 = h11[0]["comp"]  # baseline compliance
    C_allow = C0 * 1.15
    ax.plot(batches_11, comp_11, "-", color="#1565c0", linewidth=1.5,
            label="SASTO-PA (part-aware)")
    ax.plot(batches_12, comp_12, "--", color="#e65100", linewidth=1.5,
            label="SASTO-U (uniform)")
    ax.axhline(y=C_allow, color="#c62828", linewidth=2, linestyle="-",
               alpha=0.7, label=f"$C_{{allow}}$ = {C_allow:.3f} J")
    ax.set_xlabel("Batch Number")
    ax.set_ylabel("Conservative Compliance (J)")
    ax.set_title("(c) Compliance")
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig4_convergence.{ext}"))
    plt.close(fig)
    print("  OK Figure 4: Optimization convergence (SASTO-PA vs SASTO-U)")


# ═══════════════════════════════════════════════════════════════════
# Figure 5: Per-Part Volume Breakdown
# ═══════════════════════════════════════════════════════════════════
def fig5_per_part():
    parts = ["Exterior\nWall", "Interior\nWall", "Roof", "Floor"]
    original = [65240, 44388, 3746, 3498]
    optimized = [59380, 5860, 3500, 3350]
    kept_pct = [o / r * 100 for o, r in zip(optimized, original)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: Stacked bar chart
    ax = axes[0]
    x = np.arange(len(parts))
    width = 0.35
    bars1 = ax.bar(x - width / 2, original, width, label="Original ($B_0$)",
                   color="#1565c0", edgecolor="#0D3B66", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, optimized, width, label="Optimized (SASTO-PA)",
                   color="#2e7d32", edgecolor="#1B5E20", linewidth=0.8)
    ax.set_xlabel("Structural Part")
    ax.set_ylabel("Volume (voxels)")
    ax.set_title("(a) Voxel Count by Part")
    ax.set_xticks(x)
    ax.set_xticklabels(parts)
    ax.legend()
    ax.set_yscale("log")
    ax.set_ylim(1000, 200000)

    # Panel B: Retention percentage
    ax = axes[1]
    colors = ["#1565c0", "#c62828", "#2e7d32", "#e65100"]
    bars = ax.barh(parts, kept_pct, color=colors, edgecolor="white",
                   linewidth=0.8, height=0.5)
    ax.set_xlabel("Material Retained (%)")
    ax.set_title("(b) Retention by Part")
    ax.set_xlim(0, 105)
    for bar, pct in zip(bars, kept_pct):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig5_per_part.{ext}"))
    plt.close(fig)
    print("  ✓ Figure 5: Per-part volume breakdown")


# ═══════════════════════════════════════════════════════════════════
# Figure 6: Efficiency-Integrity Index Comparison
# ═══════════════════════════════════════════════════════════════════
def fig6_efficiency():
    variants = ["$B_0$\n(Baseline)", "SASTO-U\n(Uniform)", "SASTO-PA\n(Part-Aware)"]
    vol_red = [0.0, 34.3, 45.0]
    iei = [0.0, 0.242, 0.358]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: Volume reduction comparison
    ax = axes[0]
    colors = ["#bdbdbd", "#e65100", "#1565c0"]
    bars = ax.bar(variants, vol_red, color=colors, edgecolor="white",
                  linewidth=1.5, width=0.5)
    ax.set_ylabel("Volume Reduction (%)")
    ax.set_title("(a) Material Reduction")
    ax.set_ylim(0, 55)
    for bar, v in zip(bars, vol_red):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{v:.1f}%", ha="center", fontweight="bold", fontsize=11)

    # Panel B: I_EI comparison
    ax = axes[1]
    bars = ax.bar(variants, iei, color=colors, edgecolor="white",
                  linewidth=1.5, width=0.5)
    ax.set_ylabel("$\\mathcal{I}_{EI}$ (Efficiency-Integrity Index)")
    ax.set_title("(b) Efficiency-Integrity Index")
    ax.set_ylim(0, 0.45)
    for bar, v in zip(bars, iei):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", fontweight="bold", fontsize=11)

    # Add "+48%" annotation (offset to avoid overlap)
    ax.annotate("+48%", xy=(2, 0.365), xytext=(1.7, 0.415),
                fontsize=12, fontweight="bold", color="#2e7d32", ha="center",
                arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=2,
                                connectionstyle="arc3,rad=-0.15"))

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig6_efficiency.{ext}"))
    plt.close(fig)
    print("  ✓ Figure 6: Efficiency-Integrity Index comparison")


# ═══════════════════════════════════════════════════════════════════
# Figure 7: Ensemble Uncertainty Evolution
# ═══════════════════════════════════════════════════════════════════
def fig7_uncertainty():
    """
    Compute ensemble disagreement evolution from V11 history.
    The history stores conservative (μ+kσ) values. We can estimate
    uncertainty growth by tracking how VM, displacement, and compliance
    evolve relative to baseline.
    """
    v11 = load_history(V11_JSON)
    h = v11["history"]

    vol_fractions = [1.0 - e["vol_reduction"] for e in h]

    # Normalized response growth (proxy for uncertainty evolution)
    vm_baseline = h[0]["vm"]
    comp_baseline = h[0]["comp"]
    disp_baseline = h[0]["disp"]

    vm_ratio = [e["vm"] / vm_baseline for e in h]
    comp_ratio = [e["comp"] / comp_baseline for e in h]
    disp_ratio = [e["disp"] / disp_baseline for e in h]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(vol_fractions, vm_ratio, "-", color="#c62828", linewidth=1.5,
            label="VM Stress / Baseline")
    ax.plot(vol_fractions, comp_ratio, "-", color="#1565c0", linewidth=1.5,
            label="Compliance / Baseline")
    ax.plot(vol_fractions, disp_ratio, "-", color="#2e7d32", linewidth=1.5,
            label="Displacement / Baseline")

    ax.set_xlabel("Volume Fraction $\\phi = V/V_0$")
    ax.set_ylabel("Normalized Response (ratio to baseline)")
    ax.set_title("Response Evolution During Optimization (SASTO-PA)")
    ax.legend(loc="upper right")
    ax.invert_xaxis()
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)

    # Add constraint limits
    ax.axhline(y=5.0e6 / vm_baseline, color="#c62828", linestyle="--",
               alpha=0.4, label="VM allow")
    ax.axhline(y=comp_baseline * 1.15 / comp_baseline, color="#1565c0",
               linestyle="--", alpha=0.4)
    ax.text(0.82, 1.18, "$C_{allow}/C_0 = 1.15$", fontsize=9, color="#1565c0",
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.8))

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig7_uncertainty.{ext}"))
    plt.close(fig)
    print("  ✓ Figure 7: Response evolution during optimization")


# ═══════════════════════════════════════════════════════════════════
# Figure 8: Batch Size Adaptation (Trust Region Analogy)
# ═══════════════════════════════════════════════════════════════════
def fig8_batch_adaptation():
    v11 = load_history(V11_JSON)
    h = v11["history"]

    batches = [e["batch"] for e in h]
    removed = [e["removed"] for e in h]

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.bar(batches, removed, width=1, color="#1565c0", alpha=0.7,
           edgecolor="none")
    ax.set_xlabel("Batch Number")
    ax.set_ylabel("Voxels Removed per Batch")
    ax.set_title("Adaptive Batch Size During SASTO-PA Optimization")

    # Annotate phases
    ax.axvline(x=260, color="#c62828", linestyle="--", linewidth=1.5)
    ax.text(130, max(removed) * 0.9, "Phase 1: Erosion\n(260 batches)",
            ha="center", fontsize=12, color="#1565c0", fontweight="bold")
    ax.text(270, max(removed) * 0.85, "Phase 2:\nEndgame",
            ha="left", fontsize=11, color="#c62828", fontweight="bold")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig8_batch_adaptation.{ext}"))
    plt.close(fig)
    print("  ✓ Figure 8: Batch size adaptation")


# ═══════════════════════════════════════════════════════════════════
# Figure 9: Ablation Summary
# ═══════════════════════════════════════════════════════════════════
def fig9_ablation():
    configs = [
        "26-conn\n(baseline)",
        "6-conn\nuniform t=2",
        "6-conn\npart-aware",
    ]
    vol_red = [0, 34.3, 45.0]  # 26-conn produces broken meshes so 0 effective
    mesh_ok = [False, True, True]
    colors = ["#c62828", "#e65100", "#2e7d32"]
    edge_colors = ["#8B0000", "#BF360C", "#1B5E20"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(configs, vol_red, color=colors, edgecolor=edge_colors,
                  linewidth=2, width=0.5)

    # Add mesh status
    for i, (bar, ok) in enumerate(zip(bars, mesh_ok)):
        symbol = "OK: Printable" if ok else "X: Broken mesh"
        color = "#2e7d32" if ok else "#c62828"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                symbol, ha="center", fontsize=10, color=color,
                fontweight="bold")

    ax.set_ylabel("Volume Reduction (%)")
    ax.set_title("Ablation: Connectivity + Thickness Formulation")
    ax.set_ylim(0, 55)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig9_ablation.{ext}"))
    plt.close(fig)
    print("  ✓ Figure 9: Ablation summary")


# ═══════════════════════════════════════════════════════════════════
# Figure 10: k-Factor Sensitivity
# ═══════════════════════════════════════════════════════════════════
def fig10_k_sensitivity():
    k_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    # V11 actual at k=1.0; k=1.5 was ~V10 result; others estimated
    vol_red = [52, 49, 45.0, 34.0, 28]
    risk = ["High", "Moderate", "Low", "Very Low", "Minimal"]
    colors = ["#c62828", "#e65100", "#1565c0", "#2e7d32", "#4caf50"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar([str(k) for k in k_values], vol_red, color=colors,
                  edgecolor="white", linewidth=1.5, width=0.5)

    # Highlight actual measurement
    bars[2].set_edgecolor("#000000")
    bars[2].set_linewidth(2.5)
    ax.text(2, 45.0 + 1.5, "Measured", ha="center", fontsize=10,
            fontweight="bold")
    ax.text(0, 52 + 1.5, "Est.", ha="center", fontsize=9, color="#666")
    ax.text(1, 49 + 1.5, "Est.", ha="center", fontsize=9, color="#666")
    ax.text(3, 34 + 1.5, "~V10", ha="center", fontsize=9, color="#666")
    ax.text(4, 28 + 1.5, "Est.", ha="center", fontsize=9, color="#666")

    ax.set_xlabel("Uncertainty Factor $k$")
    ax.set_ylabel("Volume Reduction (%)")
    ax.set_title("Sensitivity to Uncertainty Margin Factor")
    ax.set_ylim(0, 62)

    # Risk annotation
    for i, (bar, r) in enumerate(zip(bars, risk)):
        ax.text(bar.get_x() + bar.get_width() / 2, 2,
                r, ha="center", fontsize=8, color="white",
                fontweight="bold", rotation=90, va="bottom")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig10_k_sensitivity.{ext}"))
    plt.close(fig)
    print("  ✓ Figure 10: k-factor sensitivity")


# ═══════════════════════════════════════════════════════════════════
# Figure 11: Speedup Comparison
# ═══════════════════════════════════════════════════════════════════
def fig11_speedup():
    methods = ["SIMP\n(conservative)", "SIMP\n(aggressive)", "SASTO\n(SASTO-PA)"]
    times_hr = [30.0, 5.0, 159.5 / 3600]
    colors = ["#c62828", "#e65100", "#2e7d32"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(methods, times_hr, color=colors,
                  edgecolor=["#8B0000", "#BF360C", "#1B5E20"],
                  linewidth=2, width=0.45)

    ax.set_ylabel("Wall-Clock Time (hours)")
    ax.set_title("Runtime Comparison: SIMP vs SASTO")
    ax.set_yscale("log")
    ax.set_ylim(0.02, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:g}'))

    for bar, t in zip(bars, times_hr):
        if t < 1:
            label = f"{t * 60:.1f} min"
        else:
            label = f"{t:.0f} hr"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.3,
                label, ha="center", fontsize=11, fontweight="bold")

    # Speedup annotation (placed above SASTO bar, no arrow overlap)
    ax.annotate("100\u2013700\u00d7 speedup",
                xy=(2, times_hr[2]), xytext=(0.6, 0.06),
                fontsize=12, fontweight="bold", color="#2e7d32",
                arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=2,
                                connectionstyle="arc3,rad=0.2"),
                ha="center")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig11_speedup.{ext}"))
    plt.close(fig)
    print("  ✓ Figure 11: Speedup comparison")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Generating figures to {OUT_DIR}/\n")

    fig4_convergence()
    fig5_per_part()
    fig6_efficiency()
    fig7_uncertainty()
    fig8_batch_adaptation()
    fig9_ablation()
    fig10_k_sensitivity()
    fig11_speedup()

    print(f"\nAll figures saved to {OUT_DIR}/")
    print(f"  PNG: for web/presentation")
    print(f"  PDF: for LaTeX/publication")
