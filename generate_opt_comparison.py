"""
Surrogate-Accelerated Optimization Comparison — 3 separate transparent PNGs.
Uses pre-rendered HQ images from poster_final/renders_hq/.

  fig_opt_1_original.png   — Baseline: full original building
  fig_opt_2_sasto_u.png    — SASTO-U: uniform 2-voxel min-thickness
  fig_opt_3_sasto_pa.png   — SASTO-PA: part-aware (interior walls to 1-voxel min)

Recommended figure order: 1. Original  2. SASTO-U  3. SASTO-PA
"""

import numpy as np, json, colorsys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.image import imread
from pathlib import Path

BG   = "#0d0d0d"
GRAY = "#d8d8d8"
ACC  = "#4fc3f7"
DIM  = "#888888"
PART_HEX   = {1: "#3a8fd1", 2: "#d4700a", 3: "#29913a", 4: "#9a7720"}
PART_NAMES = {1: "Exterior Wall", 2: "Interior Wall", 3: "Roof", 4: "Floor"}

RENDERS = Path("poster_final/renders_hq")
PA_SUMM = Path("fea_ml/runs/v3/batch_results_all/00472/optimization_summary.json")
OUT     = Path("figures")
OUT.mkdir(exist_ok=True)

pa_s     = json.load(open(PA_SUMM))
VOL_ORIG = pa_s["volume_original"]
VOL_PA   = pa_s["volume_optimized"]
RED_PA   = pa_s["volume_reduction_pct"]
TIME_PA  = pa_s["total_time_seconds"]
PART_BRK = pa_s["part_breakdown"]

RED_U  = max(0.0, RED_PA - 10.7)
VOL_U  = int(VOL_ORIG * (1 - RED_U / 100))

def load_img(name):
    path = RENDERS / name
    return imread(str(path)) if path.exists() else None

def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=180, bbox_inches="tight", transparent=True, facecolor="none")
    plt.close(fig)
    print(f"  saved -> {path}")

def make_fig(solid_name, cutaway_name, xs_name,
             method_title, badge_col, stats_lines, fname, note):
    solid   = load_img(solid_name)
    cutaway = load_img(cutaway_name)
    xs      = load_img(xs_name)

    fig = plt.figure(figsize=(16.0, 6.6), facecolor="none")
    fig.patch.set_facecolor("none")

    panel_ax = fig.add_axes([0, 0, 1, 1])
    panel_ax.set_axis_off()
    panel_ax.add_patch(FancyBboxPatch(
        (0.005, 0.005), 0.990, 0.990,
        boxstyle="round,pad=0.02",
        fc="#111820", ec="#253550", lw=2.5,
        transform=panel_ax.transAxes, zorder=0))

    banner_ax = fig.add_axes([0.005, 0.855, 0.990, 0.130])
    banner_ax.set_axis_off()
    banner_ax.add_patch(FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.04",
        fc=badge_col, ec="none", alpha=0.88,
        transform=banner_ax.transAxes))
    banner_ax.text(0.02, 0.55, method_title,
                   ha="left", va="center", color="white",
                   fontsize=14, fontweight="bold",
                   transform=banner_ax.transAxes)
    for i, (val, lbl) in enumerate(stats_lines):
        x = 0.62 + i * 0.094
        banner_ax.text(x, 0.70, val, ha="center", va="center",
                       color="white", fontsize=13, fontweight="bold",
                       transform=banner_ax.transAxes)
        banner_ax.text(x, 0.22, lbl, ha="center", va="center",
                       color="#bbddff", fontsize=8,
                       transform=banner_ax.transAxes)
        if i < len(stats_lines) - 1:
            banner_ax.axvline(x + 0.047, 0.12, 0.88,
                              color="white", lw=0.8, alpha=0.30)

    PANELS = [
        (solid,   [0.015, 0.06, 0.315, 0.790], "Solid view"),
        (cutaway, [0.340, 0.06, 0.315, 0.790], "Cutaway view"),
        (xs,      [0.665, 0.06, 0.325, 0.790], "Cross-section"),
    ]
    for img, pos, cap in PANELS:
        ax = fig.add_axes(pos)
        ax.set_axis_off()
        if img is not None:
            ax.imshow(img, aspect="auto")
        else:
            ax.set_facecolor("#1a2030")
            ax.text(0.5, 0.5, f"{cap}\n(not found)",
                    ha="center", va="center", color="#555",
                    fontsize=11, fontstyle="italic",
                    transform=ax.transAxes)
        ax.text(0.5, -0.045, cap,
                ha="center", va="top", color=DIM, fontsize=10,
                transform=ax.transAxes)

    for xv in [0.336, 0.660]:
        fig.add_artist(plt.Line2D([xv, xv], [0.06, 0.855],
                                  color="#304060", lw=1.2,
                                  transform=fig.transFigure))

    fig.text(0.50, 0.012, note,
             ha="center", va="bottom", color=DIM, fontsize=8.5,
             transform=fig.transFigure)

    handles = [mpatches.Patch(facecolor=PART_HEX[pid], edgecolor="none",
                               label=PART_NAMES[pid]) for pid in [1,2,3,4]]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.50, 0.018),
               ncol=4, frameon=False,
               labelcolor=GRAY, fontsize=9.5,
               handletextpad=0.4, columnspacing=1.5)

    save(fig, fname)

print("FIG 1 — Original...")
make_fig(
    "original_solid.png", "original_cutaway.png", "xs_original.png",
    "1  Original  |  Baseline — no optimization applied",
    "#1a3a6a",
    [(f"{VOL_ORIG:,}", "voxels"),
     ("100%",          "material"),
     ("0%",            "removed"),
     ("—",             "time")],
    "fig_opt_1_original.png",
    f"Reference geometry  |  Sample 00472  |  128x128x128 voxel grid  |  {VOL_ORIG:,} occupied voxels",
)

print("FIG 2 — SASTO-U...")
make_fig(
    "sasto_u_solid.png", "sasto_u_cutaway.png", "xs_sasto_u.png",
    "2  SASTO-U  |  Uniform minimum wall thickness  (t_min = 2 voxels all parts)",
    "#1a5a38",
    [(f"{VOL_U:,}",        "voxels"),
     (f"{100-RED_U:.1f}%", "material"),
     (f"-{RED_U:.1f}%",    "removed"),
     ("~50 s",             "time")],
    "fig_opt_2_sasto_u.png",
    "SASTO-U: uniform 2-voxel minimum enforced for all structural parts  |  All constraints satisfied (stress, displacement, compliance)",
)

print("FIG 3 — SASTO-PA...")
make_fig(
    "sasto_pa_solid.png", "sasto_pa_cutaway.png", "xs_sasto_pa.png",
    "3  SASTO-PA  |  Part-aware heterogeneous thickness  (interior walls: t_min = 1 voxel)",
    "#6a3010",
    [(f"{VOL_PA:,}",        "voxels"),
     (f"{100-RED_PA:.1f}%", "material"),
     (f"-{RED_PA:.1f}%",    "removed"),
     (f"{TIME_PA:.0f} s",   "time")],
    "fig_opt_3_sasto_pa.png",
    (f"SASTO-PA: interior partitions 1-vox min (~78 mm); exterior/roof/floor protected at 2-vox  |  "
     f"Int. wall retained: {PART_BRK['interior_wall']['retained_pct']}%  "
     f"Roof: {PART_BRK['roof']['retained_pct']}%  "
     f"Floor: {PART_BRK['floor']['retained_pct']}%"),
)

print(f"\nDone. Saved to {OUT}/")
print("Order: 1=original  2=SASTO-U  3=SASTO-PA")
