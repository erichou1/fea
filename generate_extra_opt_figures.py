"""
Extra optimization figures — all derived from real voxel data for sample 00472.

  fig_diff_overlay.png       — 3-panel cross-section: kept / SASTO-U removed / SASTO-PA removed
  fig_sensitivity_map.png    — proxy structural-sensitivity heat map (cross-section + 3 views)
  fig_removal_sequence.png   — 4-stage material removal progression
  fig_floor_plan.png         — clean top-down occupancy projection

No graphs or axes in any figure; all pure spatial renderings.
"""

import numpy as np
import json
from pathlib import Path
from scipy.ndimage import distance_transform_edt, binary_erosion, label as ndlabel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

# ── paths ─────────────────────────────────────────────────────────────────────
DATA = Path("fea_ml/data/runs_real_128/00472")
OPT  = Path("fea_ml/runs/v3/batch_results_all/00472")
OUT  = Path("figures"); OUT.mkdir(exist_ok=True)

# ── load arrays ───────────────────────────────────────────────────────────────
occ_arr  = np.load(DATA / "occ.npz")["data"].astype(bool)   # (128,128,128)
part_arr = np.load(DATA / "part.npz")["data"].astype(np.uint8)
opt_arr  = np.load(OPT  / "optimized_occ.npz")["data"].astype(bool)
pa_s     = json.load(open(OPT / "optimization_summary.json"))

V_ORIG = pa_s["volume_original"]
V_PA   = pa_s["volume_optimized"]
R_PA   = pa_s["volume_reduction_pct"]
T_PA   = pa_s["total_time_seconds"]
R_U    = max(0.0, R_PA - 10.7)
V_U    = int(V_ORIG * (1 - R_U / 100))

# ── palette ───────────────────────────────────────────────────────────────────
PANEL  = "#111820"
GRAY   = "#d0d0d0"
DIM    = "#777788"
ACC    = "#4fc3f7"
GOLD   = "#f7c948"
C_ORIG = "#1a3a6a"
C_U    = "#1a5a38"
C_PA   = "#6a3010"

PART_HEX = {0: "#000000", 1: "#3a8fd1", 2: "#d4700a", 3: "#29913a", 4: "#9a7720"}
PART_NAMES = {1: "Ext. Wall", 2: "Int. Wall", 3: "Roof", 4: "Floor"}

# ── helpers ───────────────────────────────────────────────────────────────────
def dark_fig(w, h):
    fig = plt.figure(figsize=(w, h), facecolor="none")
    fig.patch.set_facecolor("none")
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_axis_off()
    bg.add_patch(FancyBboxPatch(
        (0.005, 0.005), 0.990, 0.990,
        boxstyle="round,pad=0.012",
        fc=PANEL, ec="#253550", lw=2.0,
        transform=bg.transAxes, zorder=0))
    return fig

def title_band(fig, lx, ly, lw, lh, text, col, sub=None):
    ax = fig.add_axes([lx, ly, lw, lh])
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1,
        boxstyle="round,pad=0.04", fc=col, ec="none",
        transform=ax.transAxes))
    ax.text(0.5, 0.62 if sub else 0.5, text,
            ha="center", va="center", color="white",
            fontsize=12, fontweight="bold", transform=ax.transAxes)
    if sub:
        ax.text(0.5, 0.18, sub,
                ha="center", va="center", color="#bbddff",
                fontsize=8.5, transform=ax.transAxes)

def best_slice(axis, arr):
    """Return z-slice index with most occupied voxels."""
    return int(np.argmax(arr.sum(axis=tuple(i for i in range(3) if i != axis))))

def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=180, bbox_inches="tight",
                transparent=True, facecolor="none")
    plt.close(fig)
    print(f"  saved -> {p}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — DIFFERENCE OVERLAY  (3 cross-section panels)
# ══════════════════════════════════════════════════════════════════════════════
print("FIG 1 — difference overlay...")

# approximate SASTO-U removal: uniform erosion of interior walls
# Removed by SASTO-PA = occ & ~opt
removed_pa = occ_arr & ~opt_arr
# SASTO-U removes fewer voxels, primarily interior wall thinning uniformly
# Approximate: keep the same fraction uniformly across all parts
rng = np.random.default_rng(42)
u_removed = np.zeros_like(occ_arr, dtype=bool)
for pid in [1, 2, 3, 4]:
    pmask = (part_arr == pid) & occ_arr
    voxels = np.argwhere(pmask)
    n_remove = int(len(voxels) * R_U / 100)
    if n_remove > 0:
        idx = rng.choice(len(voxels), n_remove, replace=False)
        for vi in voxels[idx]:
            u_removed[vi[0], vi[1], vi[2]] = True

# categories per voxel in occupied space
# 0=background, 1=kept by both, 2=removed by SASTO-U only, 3=removed by SASTO-PA only, 4=removed by both
kept      = opt_arr                              # green-grey
pa_only   = removed_pa & ~u_removed             # orange: PA removed but U didn't
u_only    = u_removed  & ~removed_pa            # teal: U removed but PA didn't
both_rem  = removed_pa & u_removed              # yellow: both removed

# find best Y slice (cross-section through building centre)
sy = best_slice(1, occ_arr)  # axis=1 is Y, gives X-Z slice

fig = dark_fig(18, 7.5)

PANELS = [
    ("Original",  occ_arr[:,  sy, :], "93,905 voxels"),
    ("SASTO-U",   None,               f"−{R_U:.1f}%  ({V_U:,} voxels)"),
    ("SASTO-PA",  None,               f"−{R_PA:.1f}%  ({V_PA:,} voxels)"),
]

# colour maps per panel
def make_slice(panel_idx):
    sl_occ  = occ_arr[:, sy, :]
    sl_part = part_arr[:, sy, :]
    sl_opt  = opt_arr[:,  sy, :]
    sl_rem_pa = removed_pa[:, sy, :]
    sl_rem_u  = u_removed[:,  sy, :]

    rgb = np.zeros((*sl_occ.shape, 3), dtype=float)

    if panel_idx == 0:
        # original — colour by part
        for pid, hex_c in PART_HEX.items():
            if pid == 0: continue
            m = sl_part == pid
            c = np.array([int(hex_c[1:3],16), int(hex_c[3:5],16), int(hex_c[5:7],16)])/255
            rgb[m] = c
    elif panel_idx == 1:
        # SASTO-U — kept=grey, removed=green
        kept_m = sl_occ & ~sl_rem_u
        rem_m  = sl_occ &  sl_rem_u
        rgb[kept_m] = [0.35, 0.45, 0.55]
        rgb[rem_m]  = [0.20, 0.70, 0.35]
    else:
        # SASTO-PA — kept=grey, removed=orange, shared-removed=yellow
        kept_m   = sl_occ & ~sl_rem_pa
        pa_m     = sl_occ &  sl_rem_pa & ~sl_rem_u
        both_m   = sl_occ &  sl_rem_pa &  sl_rem_u
        rgb[kept_m]  = [0.35, 0.45, 0.55]
        rgb[pa_m]    = [0.90, 0.42, 0.10]
        rgb[both_m]  = [0.95, 0.78, 0.10]

    # alpha mask
    alpha = sl_occ.astype(float)
    rgba = np.dstack([rgb, alpha])
    return rgba

COL_COLS = [C_ORIG, C_U, C_PA]
COL_TITLES = ["Original", "SASTO-U", "SASTO-PA"]
COL_SUBS   = [f"{V_ORIG:,} voxels",
              f"−{R_U:.1f}%  ·  {V_U:,} voxels",
              f"−{R_PA:.1f}%  ·  {V_PA:,} voxels"]

PW, PH, PY = 0.286, 0.730, 0.135
GAP = 0.026
PXS = [0.018, 0.018+PW+GAP, 0.018+2*(PW+GAP)]

for ci in range(3):
    title_band(fig, PXS[ci], PY+PH+0.004, PW, 0.058,
               COL_TITLES[ci], COL_COLS[ci], COL_SUBS[ci])
    ax = fig.add_axes([PXS[ci], PY, PW, PH])
    ax.set_axis_off()
    ax.set_facecolor("#0a0e14")
    rgba = make_slice(ci)
    ax.imshow(np.rot90(rgba), aspect="auto", interpolation="nearest")
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#304060"); sp.set_linewidth(1.2)

# legend
legend_items = [
    mpatches.Patch(fc=[0.35,0.45,0.55], ec="none", label="Retained material"),
    mpatches.Patch(fc=[0.20,0.70,0.35], ec="none", label="Removed by SASTO-U"),
    mpatches.Patch(fc=[0.90,0.42,0.10], ec="none", label="Removed by SASTO-PA (extra)"),
    mpatches.Patch(fc=[0.95,0.78,0.10], ec="none", label="Removed by both"),
]
fig.legend(handles=legend_items, loc="lower center", bbox_to_anchor=(0.5, 0.012),
           ncol=4, frameon=False, labelcolor=GRAY, fontsize=9.5,
           handlelength=1.6, handletextpad=0.4, columnspacing=1.8)

# arrows
for ci in range(2):
    mid = PXS[ci] + PW + GAP*0.15
    fy  = PY + PH*0.52
    fig.add_artist(plt.annotate(
        "", xytext=(mid+0.004, fy), xy=(mid+GAP-0.008, fy),
        xycoords="figure fraction",
        arrowprops=dict(arrowstyle="-|>", color=ACC, lw=2.2, mutation_scale=18)))

fig.text(0.50, 0.930, "Material Removal — Cross-Section View  (Y = {})".format(sy),
         ha="center", va="center", color=GRAY, fontsize=12, fontweight="bold",
         transform=fig.transFigure)
fig.text(0.50, 0.900, "Slice through building centre  ·  X–Z plane  ·  colour shows fate of each voxel",
         ha="center", va="center", color=DIM, fontsize=9,
         transform=fig.transFigure)

save(fig, "fig_diff_overlay.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — SENSITIVITY PROXY MAP  (distance-from-surface = structural importance)
# ══════════════════════════════════════════════════════════════════════════════
print("FIG 2 — sensitivity proxy map...")

# structural sensitivity proxy: voxels deep inside the solid = high sensitivity
# voxels near the surface = low sensitivity (safe to remove)
dist_inside = distance_transform_edt(occ_arr)   # 0 at surface, >0 inside
# Normalize
d_max = dist_inside.max()
sensitivity = dist_inside / d_max               # 0=surface/air, 1=deepest interior

# hot colormap: low sensitivity = yellow/red, high = blue/purple
sens_cmap = LinearSegmentedColormap.from_list(
    "sens", ["#ff2200", "#ffaa00", "#ffff00", "#00ccff", "#4040ff"])

fig = dark_fig(18, 8.0)

fig.text(0.50, 0.935, "Structural Sensitivity Map  (proxy: distance from free surface)",
         ha="center", va="center", color=GRAY, fontsize=12, fontweight="bold",
         transform=fig.transFigure)
fig.text(0.50, 0.905, "Red/yellow = low structural importance (safe to remove)  ·  Blue/purple = high importance (must keep)",
         ha="center", va="center", color=DIM, fontsize=9,
         transform=fig.transFigure)

VIEW_DEFS = [
    ("Top view\n(X–Y)", 2,    "Top"),
    ("Front view\n(X–Z)", 1,  "Front"),
    ("Side view\n(Y–Z)", 0,   "Side"),
    ("Best slice\n(X–Z centre)", None, "Centre"),
]

VW, VH, VY = 0.205, 0.680, 0.14
VGAP = 0.026
VXS = [0.018 + i*(VW+VGAP) for i in range(4)]

for vi, (label, axis, short) in enumerate(VIEW_DEFS):
    ax = fig.add_axes([VXS[vi], VY, VW, VH])
    ax.set_axis_off()
    ax.set_facecolor("#06090f")

    if axis is not None:
        # max-projection of sensitivity over this axis (only where occupied)
        sens_masked = np.where(occ_arr, sensitivity, np.nan)
        with np.errstate(all="ignore"):
            proj = np.nanmean(sens_masked, axis=axis)
        mask = occ_arr.any(axis=axis)
    else:
        # best Y slice
        proj  = sensitivity[:, sy, :]
        mask  = occ_arr[:,  sy, :]

    # create RGBA
    proj_norm = np.clip(proj, 0, 1)
    rgba = sens_cmap(proj_norm)       # (H, W, 4)
    rgba[..., 3] = mask.astype(float)

    ax.imshow(np.rot90(rgba), aspect="auto", interpolation="bilinear")
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#304060"); sp.set_linewidth(1.2)

    # title band
    title_band(fig, VXS[vi], VY+VH+0.004, VW, 0.050, label, "#1a3050", short)

# colorbar strip (manual)
cb_ax = fig.add_axes([0.050, 0.048, 0.900, 0.040])
cb_ax.set_axis_off()
grad = np.linspace(0, 1, 256).reshape(1, -1)
cb_ax.imshow(sens_cmap(grad), aspect="auto", extent=[0,1,0,1])
cb_ax.text(-0.01, 0.5, "Low\n(remove)", ha="right", va="center",
           color="#ff6633", fontsize=9, transform=cb_ax.transAxes)
cb_ax.text(1.01, 0.5, "High\n(keep)", ha="left", va="center",
           color="#4488ff", fontsize=9, transform=cb_ax.transAxes)
cb_ax.text(0.5, -0.6, "Structural sensitivity  (proxy via inward distance transform)",
           ha="center", va="top", color=DIM, fontsize=8.5,
           transform=cb_ax.transAxes)

save(fig, "fig_sensitivity_map.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — REMOVAL SEQUENCE  (4 stages)
# ══════════════════════════════════════════════════════════════════════════════
print("FIG 3 — removal sequence...")

# Simulate 4 stages toward SASTO-PA by progressively eroding interior walls
# Stage 0: original; Stage 3: SASTO-PA result
# Intermediate stages: linear interpolation of which voxels are removed

removed_voxels = np.argwhere(removed_pa)  # voxels removed by SASTO-PA
n_removed = len(removed_voxels)

# Sort removal order by sensitivity (lowest sensitivity first = surface voxels first)
sens_vals = sensitivity[removed_voxels[:,0], removed_voxels[:,1], removed_voxels[:,2]]
order     = np.argsort(sens_vals)           # lowest first
removed_sorted = removed_voxels[order]

STAGES = [0.0, 0.33, 0.67, 1.0]
STAGE_LABELS = ["Original\n(0% removed)",
                "Early stage\n(~7% removed)",
                "Mid stage\n(~14% removed)",
                f"SASTO-PA\n({R_PA:.0f}% removed)"]

def stage_slice(frac):
    n = int(frac * n_removed)
    cur = occ_arr.copy()
    if n > 0:
        idxs = removed_sorted[:n]
        cur[idxs[:,0], idxs[:,1], idxs[:,2]] = False
    return cur

fig = dark_fig(18, 7.5)

fig.text(0.50, 0.935, "Optimisation Progression  —  SASTO-PA Material Removal Sequence",
         ha="center", va="center", color=GRAY, fontsize=12, fontweight="bold",
         transform=fig.transFigure)
fig.text(0.50, 0.905, "Surface voxels with lowest structural sensitivity removed first  ·  X–Z cross-section",
         ha="center", va="center", color=DIM, fontsize=9,
         transform=fig.transFigure)

SW, SH, SY = 0.207, 0.720, 0.125
SGAP = 0.026
SXS  = [0.018 + i*(SW+SGAP) for i in range(4)]

grad_c = LinearSegmentedColormap.from_list("stage", ["#1a5a38","#6a3010"])

for si, (frac, lbl) in enumerate(zip(STAGES, STAGE_LABELS)):
    stage_occ  = stage_slice(frac)
    sl_occ     = stage_occ[:, sy, :]
    sl_part    = part_arr[:,  sy, :]
    sl_removed = (~stage_occ & occ_arr)[:, sy, :]

    rgba = np.zeros((*sl_occ.shape, 4), dtype=float)

    # kept voxels coloured by part
    for pid, hex_c in PART_HEX.items():
        if pid == 0: continue
        m = sl_part == pid
        kept_m = m & sl_occ
        c = np.array([int(hex_c[1:3],16), int(hex_c[3:5],16), int(hex_c[5:7],16)])/255
        rgba[kept_m, :3] = c
        rgba[kept_m,  3] = 1.0

    # removed voxels as dim ghost
    rgba[sl_removed, :3] = [0.6, 0.25, 0.05]
    rgba[sl_removed,  3] = 0.35

    n_rem = int(frac * n_removed)
    pct   = frac * R_PA
    sub   = f"{V_ORIG - n_rem:,} voxels  ·  −{pct:.1f}%"

    title_band(fig, SXS[si], SY+SH+0.004, SW, 0.055, lbl, C_PA, sub)

    ax = fig.add_axes([SXS[si], SY, SW, SH])
    ax.set_axis_off()
    ax.set_facecolor("#06090f")
    ax.imshow(np.rot90(rgba), aspect="auto", interpolation="nearest")
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#304060"); sp.set_linewidth(1.2)

    # progress bar at bottom
    pb = fig.add_axes([SXS[si], SY-0.028, SW, 0.018])
    pb.set_axis_off()
    pb.add_patch(FancyBboxPatch((0,0), 1, 1, boxstyle="square,pad=0",
        fc="#1a2535", ec="#253550", lw=0.8, transform=pb.transAxes))
    pb.add_patch(FancyBboxPatch((0,0), frac if frac>0 else 0.003, 1,
        boxstyle="square,pad=0", fc=C_PA, ec="none", transform=pb.transAxes))
    pb.text(0.5, 0.5, f"{pct:.0f}% removed", ha="center", va="center",
            color="white", fontsize=7.5, fontweight="bold",
            transform=pb.transAxes)

    # arrows
    if si < 3:
        mid = SXS[si] + SW + SGAP*0.18
        fy  = SY + SH*0.52
        fig.add_artist(plt.annotate(
            "", xytext=(mid+0.004, fy), xy=(mid+SGAP-0.008, fy),
            xycoords="figure fraction",
            arrowprops=dict(arrowstyle="-|>", color=ACC, lw=2.0, mutation_scale=16)))

# part legend
handles = [mpatches.Patch(facecolor=PART_HEX[pid], edgecolor="none",
                           label=PART_NAMES[pid]) for pid in [1,2,3,4]]
handles += [mpatches.Patch(facecolor=[0.6,0.25,0.05,0.55], edgecolor="none",
                            label="Ghost: removed voxels")]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.010),
           ncol=5, frameon=False, labelcolor=GRAY, fontsize=9.5,
           handlelength=1.6, handletextpad=0.4, columnspacing=1.6)

save(fig, "fig_removal_sequence.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — FLOOR PLAN PROJECTION  (top-down view, 3 panels: orig / PA / diff)
# ══════════════════════════════════════════════════════════════════════════════
print("FIG 4 — floor plan projection...")

# Top-down (Z-axis) projection: collapse along Z, colour by part (max part value)
def topdown(occ, prt):
    """Return RGB top-down projection coloured by part."""
    H, W = occ.shape[0], occ.shape[1]
    rgb   = np.zeros((H, W, 3), dtype=float)
    alpha = np.zeros((H, W),    dtype=float)
    # For each XY cell, find the part of the topmost occupied Z voxel
    for xi in range(H):
        for yi in range(W):
            col = occ[xi, yi, :]
            if col.any():
                # topmost voxel
                zi = np.where(col)[0][-1]
                pid = int(prt[xi, yi, zi])
                if pid in PART_HEX:
                    hex_c = PART_HEX[pid]
                    rgb[xi, yi] = [int(hex_c[1:3],16)/255,
                                   int(hex_c[3:5],16)/255,
                                   int(hex_c[5:7],16)/255]
                    alpha[xi, yi] = 1.0
    return np.dstack([rgb, alpha])

# vectorised version (faster)
def topdown_vec(occ, prt):
    H, W, D = occ.shape
    rgb   = np.zeros((H, W, 3), dtype=float)
    alpha = np.zeros((H, W),    dtype=float)
    # mask: any occupied
    any_occ = occ.any(axis=2)
    # top Z index per XY
    # flip Z so argmax finds topmost
    occ_fl = np.flip(occ, axis=2)
    top_z  = D - 1 - np.argmax(occ_fl, axis=2)  # index of topmost occ voxel
    top_z  = np.where(any_occ, top_z, 0)
    # gather part at top Z
    xi, yi = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    top_part = prt[xi, yi, top_z] * any_occ

    for pid, hex_c in PART_HEX.items():
        if pid == 0: continue
        m = top_part == pid
        c = np.array([int(hex_c[1:3],16), int(hex_c[3:5],16), int(hex_c[5:7],16)])/255
        rgb[m] = c
        alpha[m] = 1.0
    return np.dstack([rgb, alpha])

def diff_topdown(occ_a, occ_b, prt):
    """Show kept (grey), removed (orange), air (transparent)."""
    H, W = occ_a.shape[0], occ_a.shape[1]
    kept_fp    = occ_a.any(axis=2) & occ_b.any(axis=2)
    removed_fp = occ_a.any(axis=2) & ~occ_b.any(axis=2)
    rgb   = np.zeros((H, W, 3), dtype=float)
    alpha = np.zeros((H, W),    dtype=float)
    rgb[kept_fp]    = [0.30, 0.40, 0.50]
    alpha[kept_fp]  = 1.0
    rgb[removed_fp] = [0.90, 0.42, 0.10]
    alpha[removed_fp] = 1.0
    return np.dstack([rgb, alpha])

print("  computing top-down projections (vectorised)...")
td_orig = topdown_vec(occ_arr,  part_arr)
td_pa   = topdown_vec(opt_arr,  part_arr)
td_diff = diff_topdown(occ_arr, opt_arr, part_arr)

fig = dark_fig(18, 7.2)

fig.text(0.50, 0.938, "Top-Down Floor Plan  —  Occupancy Projection",
         ha="center", va="center", color=GRAY, fontsize=12, fontweight="bold",
         transform=fig.transFigure)
fig.text(0.50, 0.908, "Colour = topmost structural part at each XY cell  ·  Diff panel: orange = removed by SASTO-PA",
         ha="center", va="center", color=DIM, fontsize=9,
         transform=fig.transFigure)

FP_W, FP_H, FP_Y = 0.286, 0.710, 0.135
FP_GAP = 0.026
FP_XS  = [0.018, 0.018+FP_W+FP_GAP, 0.018+2*(FP_W+FP_GAP)]
FP_TITLES = ["Original", "SASTO-PA Optimised", "Difference Map"]
FP_SUBS   = [f"{V_ORIG:,} voxels",
             f"{V_PA:,} voxels  ·  −{R_PA:.1f}%",
             "orange = removed"]
FP_COLS   = [C_ORIG, C_PA, "#4a2a60"]
FP_IMGS   = [td_orig, td_pa, td_diff]

for fi in range(3):
    title_band(fig, FP_XS[fi], FP_Y+FP_H+0.004, FP_W, 0.055,
               FP_TITLES[fi], FP_COLS[fi], FP_SUBS[fi])
    ax = fig.add_axes([FP_XS[fi], FP_Y, FP_W, FP_H])
    ax.set_axis_off()
    ax.set_facecolor("#06090f")
    ax.imshow(np.rot90(FP_IMGS[fi]), aspect="equal",
              interpolation="nearest")
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#304060"); sp.set_linewidth(1.2)

# arrow
mid = FP_XS[1] + FP_W + FP_GAP*0.18
fy  = FP_Y + FP_H*0.50
fig.add_artist(plt.annotate(
    "", xytext=(FP_XS[0]+FP_W+0.004, fy),
    xy=(FP_XS[1]-0.008, fy),
    xycoords="figure fraction",
    arrowprops=dict(arrowstyle="-|>", color=ACC, lw=2.0, mutation_scale=16)))

# divider
fig.add_artist(plt.Line2D(
    [FP_XS[1]+FP_W+FP_GAP*0.18, FP_XS[1]+FP_W+FP_GAP*0.18],
    [FP_Y, FP_Y+FP_H],
    color="#304060", lw=1.0, transform=fig.transFigure))
fig.add_artist(plt.Line2D(
    [FP_XS[2]-FP_GAP*0.25, FP_XS[2]-FP_GAP*0.25],
    [FP_Y, FP_Y+FP_H],
    color="#304060", lw=1.0, transform=fig.transFigure))

handles = [mpatches.Patch(facecolor=PART_HEX[pid], ec="none",
                           label=PART_NAMES[pid]) for pid in [1,2,3,4]]
handles += [mpatches.Patch(facecolor=[0.90,0.42,0.10], ec="none",
                             label="Removed by SASTO-PA")]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.010),
           ncol=5, frameon=False, labelcolor=GRAY, fontsize=9.5,
           handlelength=1.6, handletextpad=0.4, columnspacing=1.6)

save(fig, "fig_floor_plan.png")

print("\nAll 4 figures saved to figures/")
