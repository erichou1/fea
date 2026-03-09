"""
Generate fig_type_comparison-style figures for 5 diverse house designs.
Each figure: 4 rows (Front / Isometric / Top / Interior Cutaway)
             3 cols (Original | SASTO-U | SASTO-PA)
Saves to results_figures/fig_house_{rank}_{sid}.png
"""
import json, sys
import numpy as np
from pathlib import Path

# ── import helpers from render_figures ---------------------------------
sys.path.insert(0, str(Path(__file__).parent))
import render_figures as rf

OUT = Path("results_figures")
OUT.mkdir(exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BG_RGB = np.array(rf.BG_COLOR[:3]) / 255.0
GRAY   = "#1c2233"

# ── pick 5 diverse designs from the batch ------------------------------
print("Scanning batch results...")
samples = []
for d in sorted(rf.BATCH_DIR.iterdir()):
    sp = d / "optimization_summary.json"
    op = d / "optimized_occ.npz"
    if not (sp.exists() and op.exists()):
        continue
    try:
        s = json.load(open(sp))
    except Exception:
        continue
    if not s.get("constraints_satisfied"):
        continue
    sid = s.get("sample_id")
    base_path = rf.DATA_DIR / sid / "occ.npz"
    if not base_path.exists():
        continue
    samples.append((s["volume_reduction_pct"], sid, d))

samples.sort(key=lambda x: x[0], reverse=True)
n = len(samples)
print(f"  Found {n} feasible designs")

# 5 evenly-spaced across the full spectrum (skip rank-0 — often extreme)
indices = [max(1, int(n * f)) for f in [0.05, 0.25, 0.50, 0.75, 0.95]]
selected = [samples[min(i, n-1)] for i in indices]

# ── render each house ---------------------------------------------------
for rank_pos, (red, sid, batch_dir) in enumerate(selected, 1):
    print(f"\n[{rank_pos}/5]  {sid}  ({red:.1f}% reduction)")

    base_occ  = np.load(str(rf.DATA_DIR / sid / "occ.npz"))["data"]
    opt_occ   = np.load(str(batch_dir / "optimized_occ.npz"))["data"]

    part_path = rf.DATA_DIR / sid / "part.npz"
    fp_path   = batch_dir / "fixed_part.npz"
    if fp_path.exists():
        part = np.load(str(fp_path))["data"]
    elif part_path.exists():
        part = np.load(str(part_path))["data"]
    else:
        part = np.zeros_like(base_occ, dtype=np.int32)

    # ── load SASTO-U result -----------------------------------------------
    u_path = batch_dir / "sasto_u" / "optimized_occ.npz"
    has_u  = u_path.exists()
    if has_u:
        u_occ = np.load(str(u_path))["data"]
        u_summ_path = batch_dir / "sasto_u" / "optimization_summary.json"
        u_red = json.load(open(u_summ_path)).get("volume_reduction_pct", 0.0) if u_summ_path.exists() else 0.0
    else:
        print("  WARNING: no SASTO-U result found")

    print("  Building meshes...")
    orig_mesh = rf.build_colored_mesh(base_occ, part)
    opt_mesh  = rf.build_colored_mesh(opt_occ,  part)
    orig_cut  = rf.build_cutaway(base_occ, part)
    opt_cut   = rf.build_cutaway(opt_occ,  part)
    if has_u:
        u_mesh = rf.build_colored_mesh(u_occ, part)
        u_cut  = rf.build_cutaway(u_occ, part)

    if orig_mesh is None or opt_mesh is None:
        print("  SKIP — mesh build failed")
        continue

    poses, _ = rf.compute_camera_poses(orig_mesh)
    viewpoints  = ["front", "isometric", "top", "cutaway_front"]
    view_labels = ["Front", "Isometric", "Top", "Interior Cutaway"]

    n_base = int(base_occ.sum())
    n_opt  = int(opt_occ.sum())

    if has_u:
        n_u = int(u_occ.sum())
        col_titles = [
            "Original",
            f"SASTO-U  (−{u_red:.1f}%,  −{n_base-n_u:,} vox)",
            f"SASTO-PA  (−{red:.1f}%,  −{n_base-n_opt:,} vox)",
        ]
        meshes     = [orig_mesh, u_mesh,  opt_mesh]
        cut_meshes = [orig_cut,  u_cut,   opt_cut]
        n_cols, fig_w = 3, 15
    else:
        col_titles = [
            "Original",
            f"SASTO-PA  (−{red:.1f}%,  −{n_base-n_opt:,} vox)",
        ]
        meshes     = [orig_mesh,  opt_mesh]
        cut_meshes = [orig_cut,   opt_cut]
        n_cols, fig_w = 2, 10

    fig, axes = plt.subplots(4, n_cols, figsize=(fig_w, 19), facecolor=BG_RGB)
    for ax in axes.flat:
        ax.set_facecolor(BG_RGB); ax.axis("off")

    for col, (mesh, cut, title) in enumerate(zip(meshes, cut_meshes, col_titles)):
        # Rows 0-2: standard views
        for row, vp in enumerate(viewpoints[:3]):
            print(f"    {title}  {view_labels[row]}...")
            img = rf._pad_to_uniform(rf.trim_whitespace(
                      rf.render_mesh(mesh, poses[vp])))
            axes[row, col].imshow(img)
        # Row 3: cutaway
        if cut is not None:
            print(f"    {title}  Cutaway...")
            img = rf._pad_to_uniform(rf.trim_whitespace(
                      rf.render_mesh(cut, poses["cutaway_front"])))
            axes[3, col].imshow(img)
        axes[0, col].set_title(title, fontsize=13, fontweight="bold",
                               color=GRAY, pad=8)

    # Row labels
    for row, lbl in enumerate(view_labels):
        axes[row, 0].set_ylabel(lbl, fontsize=12, fontweight="bold",
                                rotation=90, labelpad=14, color=GRAY)
        axes[row, 0].yaxis.label.set_visible(True)
        axes[row, 0].tick_params(left=False, labelleft=False)

    rf.add_part_legend(fig)
    fig.suptitle(f"House  {sid}   —   Original vs SASTO-U vs SASTO-PA",
                 fontsize=15, fontweight="bold", color=GRAY, y=1.005)
    plt.subplots_adjust(hspace=0.04, wspace=0.02,
                        top=0.97, bottom=0.06, left=0.07, right=0.98)

    out = OUT / f"fig_house_{rank_pos:02d}_{sid}.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight",
                facecolor=BG_RGB, edgecolor="none")
    plt.close(fig)
    kb = out.stat().st_size // 1024
    print(f"  Saved  {out.name}  ({kb} KB)")

print("\nDone — 5 figures in results_figures/")
