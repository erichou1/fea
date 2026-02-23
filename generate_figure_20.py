#!/usr/bin/env python3
"""
Generate Figure 20: Multi-Geometry Optimization Results (N=20).
Creates a comprehensive visualization of the batch optimization results.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_RESULTS = os.path.join(
    os.path.dirname(__file__),
    "fea_ml", "runs", "v3", "batch_results", "aggregate_results.json",
)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def main():
    data = json.load(open(BATCH_RESULTS))
    samples = data["per_sample"]

    # Sort by original volume
    samples.sort(key=lambda s: s["volume_original"])
    sids = [s["sample_id"] for s in samples]
    vol_orig = [s["volume_original"] for s in samples]
    vol_red = [s["volume_reduction_pct"] for s in samples]
    ok = [s["constraints_satisfied"] for s in samples]
    runtimes = [s["runtime_s"] for s in samples]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Panel A: Volume reduction per sample ──────────────────
    ax = axes[0, 0]
    colors = ["#2e7d32" if c else "#c62828" for c in ok]
    bars = ax.bar(range(len(sids)), vol_red, color=colors, edgecolor="white",
                  linewidth=0.8)
    ax.set_xticks(range(len(sids)))
    ax.set_xticklabels(sids, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Volume Reduction (%)")
    ax.set_title("(a) Volume Reduction per Sample")
    ax.axhline(y=0, color="black", linewidth=0.5)

    # Legend
    ok_patch = mpatches.Patch(color="#2e7d32", label="Constraints OK (7)")
    viol_patch = mpatches.Patch(color="#c62828", label="Constraints Violated (13)")
    ax.legend(handles=[ok_patch, viol_patch], loc="upper left", fontsize=9)

    # Mean line for OK models
    ok_reds = [v for v, o in zip(vol_red, ok) if o]
    ax.axhline(y=np.mean(ok_reds), color="#2e7d32", linewidth=1.5,
               linestyle="--", alpha=0.7, label=f"OK mean: {np.mean(ok_reds):.1f}%")

    # ── Panel B: Volume vs Reduction scatter ──────────────────
    ax = axes[0, 1]
    for s in samples:
        c = "#2e7d32" if s["constraints_satisfied"] else "#c62828"
        m = "o" if s["constraints_satisfied"] else "x"
        sz = 80 if s["constraints_satisfied"] else 50
        ax.scatter(s["volume_original"] / 1000, s["volume_reduction_pct"],
                   c=c, marker=m, s=sz,
                   edgecolors="white" if s["constraints_satisfied"] else "none",
                   linewidths=0.5, zorder=3)
    ax.set_xlabel("Original Volume (x1000 voxels)")
    ax.set_ylabel("Volume Reduction (%)")
    ax.set_title("(b) Reduction vs. Geometry Size")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend(handles=[ok_patch, viol_patch], loc="upper right", fontsize=9)

    # ── Panel C: Runtime distribution ─────────────────────────
    ax = axes[1, 0]
    ok_times = [r for r, o in zip(runtimes, ok) if o]
    viol_times = [r for r, o in zip(runtimes, ok) if not o]
    ax.hist([ok_times, viol_times], bins=10, color=["#2e7d32", "#c62828"],
            label=["Constraints OK", "Violated"], edgecolor="white",
            stacked=True)
    ax.set_xlabel("Runtime (seconds)")
    ax.set_ylabel("Count")
    ax.set_title("(c) Runtime Distribution")
    ax.legend(loc="upper right", fontsize=9)
    ax.axvline(x=np.mean(runtimes), color="black", linestyle="--",
               linewidth=1.5, label=f"Mean: {np.mean(runtimes):.0f}s")

    # ── Panel D: Per-part retention (OK models only) ──────────
    ax = axes[1, 1]
    # Load per-part data from individual summaries
    batch_dir = os.path.join(os.path.dirname(__file__),
                             "fea_ml", "runs", "v3", "batch_results")
    parts_data = {"exterior_wall": [], "interior_wall": [], "roof": [], "floor": []}
    for s in samples:
        if not s["constraints_satisfied"]:
            continue
        summary_path = os.path.join(batch_dir, s["sample_id"],
                                    "optimization_summary.json")
        if os.path.exists(summary_path):
            r = json.load(open(summary_path))
            for part in parts_data:
                parts_data[part].append(r["part_breakdown"][part]["retained_pct"])

    part_names = ["Exterior\nWall", "Interior\nWall", "Roof", "Floor"]
    part_keys = ["exterior_wall", "interior_wall", "roof", "floor"]
    means = [np.mean(parts_data[k]) for k in part_keys]
    stds = [np.std(parts_data[k]) for k in part_keys]
    x = np.arange(len(part_names))
    part_colors = ["#1565c0", "#c62828", "#2e7d32", "#e65100"]
    bars = ax.bar(x, means, yerr=stds, color=part_colors, edgecolor="white",
                  linewidth=0.8, capsize=5, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(part_names)
    ax.set_ylabel("Material Retained (%)")
    ax.set_title("(d) Per-Part Retention (7 Constraint-OK Models)")
    ax.set_ylim(0, 115)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 2,
                f"{m:.0f}%", ha="center", fontsize=10, fontweight="bold")

    plt.suptitle(
        "Figure 20: Multi-Geometry SASTO-PA Optimization Results (N = 20)",
        fontsize=14, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig20_multi_geometry.{ext}"))
    plt.close(fig)
    print("  OK Figure 20: Multi-geometry optimization results")


if __name__ == "__main__":
    main()
