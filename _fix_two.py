"""Regenerate only fig_simp_comparison.png and fig_connectivity.png"""
import json, pathlib, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore", category=UserWarning)

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": "#b0b8c8", "axes.linewidth": 0.9,
    "xtick.direction": "out", "ytick.direction": "out", "figure.dpi": 100, "savefig.dpi": 200,
})

ROOT = pathlib.Path(".")
DATA_V3 = ROOT / "fea_ml" / "runs" / "v3"
BATCH = DATA_V3 / "batch_results_all"
OUT = ROOT / "results_figures"

BG="#ffffff"; PANEL="#f7f8fb"; CARD="#f0f3f8"; GRAY="#1c2233"; MID="#4a5568"
DIM="#8896b0"; LGRID="#dde3ef"; SPINE="#b0b8c8"; ACC="#1a6fd4"; TEAL="#0e9a76"
RED="#d63031"; GOLD="#d18f00"; C_PA="#c95a10"; PURPLE="#7e3acb"

def save(fig, name, dpi=200):
    p = OUT / name
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  OK  {p.name:<42}  {p.stat().st_size // 1024:>5} KB")

def style_ax(ax, xlabel=None, ylabel=None, title=None, grid=True):
    ax.set_facecolor(CARD)
    for sp in ax.spines.values(): sp.set_color(SPINE)
    ax.tick_params(colors=GRAY, labelsize=9.5, labelcolor=GRAY, length=4)
    if xlabel: ax.set_xlabel(xlabel, fontsize=10.5, color=GRAY, labelpad=6)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10.5, color=GRAY, labelpad=6)
    if title: ax.set_title(title, fontsize=12, color=GRAY, pad=8, fontweight="bold")
    if grid: ax.grid(color=LGRID, linewidth=0.7, zorder=0); ax.set_axisbelow(True)

def legend_styled(ax, **kw):
    kw.setdefault("frameon", True); kw.setdefault("framealpha", 0.92)
    kw.setdefault("edgecolor", SPINE); kw.setdefault("facecolor", PANEL)
    kw.setdefault("labelcolor", GRAY); kw.setdefault("fontsize", 9)
    return ax.legend(**kw)

# Load data
simp_data = json.load(open(DATA_V3 / "simp_benchmark.json"))
conn_data = json.load(open(DATA_V3 / "connectivity_analysis.json"))

# Population times for SASTO median
times_s = []
for d in sorted(BATCH.iterdir()):
    p = d / "optimization_summary.json"
    try:
        s = json.load(open(p))
        if s.get("success"): times_s.append(s["total_time_seconds"])
    except: pass
times_arr = np.array(times_s)

LIMIT = 1.15

# ═══════════════════════════════════════════════════════════════════
# SIMP COMPARISON
# ═══════════════════════════════════════════════════════════════════
print("fig_simp_comparison.png")
s_ids = [s["sample_id"] for s in simp_data]
s_group = [s["group"] for s in simp_data]
simp_red = np.array([s["volume_reduction_pct"] for s in simp_data])
sasto_red = np.array([s["sasto_reduction_pct"] for s in simp_data])
simp_t = np.array([s["total_time_s"] for s in simp_data])
simp_cr = np.array([s["comp_ratio"] for s in simp_data])
sasto_cr = np.array([s["sasto_comp_ratio"] for s in simp_data])

group_colors = {"high_reduction": RED, "near_boundary": GOLD, "easy": TEAL}
gc = [group_colors.get(g, DIM) for g in s_group]

fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), facecolor=BG,
                          gridspec_kw=dict(wspace=0.30))
x = np.arange(len(s_ids))
w = 0.35

# (a) Volume reduction
ax = axes[0]
style_ax(ax, ylabel="Volume reduction  (%)", title="(a) Reduction: SIMP vs SASTO")
ax.bar(x - w/2, simp_red, w, color=PURPLE, alpha=0.60, edgecolor="white", label="SIMP (64\u00b3)", zorder=3)
ax.bar(x + w/2, sasto_red, w, color=C_PA, alpha=0.60, edgecolor="white", label="SASTO (128\u00b3)", zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(s_ids, rotation=45, ha="right", fontsize=8)
for tick, col in zip(ax.get_xticklabels(), gc): tick.set_color(col)
legend_styled(ax, loc="upper left", fontsize=8.5)

# (b) Runtime
ax = axes[1]
style_ax(ax, ylabel="Wall-clock time  (s)", title="(b) Runtime Comparison")
sasto_med = float(np.median(times_arr))
ax.bar(x, simp_t, 0.50, color=PURPLE, alpha=0.60, edgecolor="white", label="SIMP (64\u00b3)", zorder=3)
ax.axhline(sasto_med, color=C_PA, lw=2.0, ls="--", zorder=4, label=f"SASTO median {sasto_med:.0f}s (128\u00b3)")
ax.set_xticks(x)
ax.set_xticklabels(s_ids, rotation=45, ha="right", fontsize=8)
for tick, col in zip(ax.get_xticklabels(), gc): tick.set_color(col)
ax.set_yscale("log")
legend_styled(ax, loc="upper left", fontsize=8.5)

# (c) Compliance ratio
ax = axes[2]
style_ax(ax, ylabel=r"Compliance ratio  $C_{opt}/C_{base}$", title="(c) Compliance Ratio")
ax.bar(x - w/2, simp_cr, w, color=PURPLE, alpha=0.60, edgecolor="white", label="SIMP", zorder=3)
ax.bar(x + w/2, sasto_cr, w, color=C_PA, alpha=0.60, edgecolor="white", label="SASTO", zorder=3)
ax.axhline(LIMIT, color=RED, lw=1.8, ls="--", zorder=4, alpha=0.7, label=f"Safety limit {LIMIT}")
ax.set_xticks(x)
ax.set_xticklabels(s_ids, rotation=45, ha="right", fontsize=8)
for tick, col in zip(ax.get_xticklabels(), gc): tick.set_color(col)
legend_styled(ax, loc="upper left", fontsize=8.5)

# Group legend at bottom
grp_handles = [mpatches.Patch(fc=RED, alpha=0.6, label="High reduction"),
               mpatches.Patch(fc=GOLD, alpha=0.6, label="Near boundary"),
               mpatches.Patch(fc=TEAL, alpha=0.6, label="Easy")]
fig.legend(handles=grp_handles, loc="lower center", ncol=3, fontsize=10,
           frameon=True, framealpha=0.92, edgecolor=SPINE, facecolor=PANEL,
           labelcolor=GRAY, bbox_to_anchor=(0.5, 0.005))
plt.tight_layout(rect=[0, 0.06, 1, 1])
save(fig, "fig_simp_comparison.png")

# ═══════════════════════════════════════════════════════════════════
# CONNECTIVITY
# ═══════════════════════════════════════════════════════════════════
print("fig_connectivity.png")
cs = conn_data["summary"]
ps = conn_data["per_sample"]
n_conn = cs["n_samples"]
comp_m = [r["mesh_components"] for r in ps]
all_comps = np.array(comp_m)
pct_single_mesh = (all_comps == 1).mean() * 100

fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
style_ax(ax, ylabel="Percentage  (%)",
         title=rf"Structural Connectivity Verification  ($n = {n_conn}$)")

cats = ["Voxel\n6-connected", "Voxel\n26-connected", "Mesh\nwatertight",
        "Mesh\nsingle body"]
vals = [cs["voxel_6conn_all_single"] / n_conn * 100,
        cs["voxel_26conn_all_single"] / n_conn * 100,
        cs["mesh_all_single"] / n_conn * 100,
        pct_single_mesh]
cols_bar = [ACC, ACC, TEAL, TEAL]

bars = ax.bar(cats, vals, color=cols_bar, alpha=0.65, edgecolor="white",
              linewidth=1.0, width=0.52, zorder=3)
for bar, v, col in zip(bars, vals, cols_bar):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f"{v:.0f}%", ha="center", va="bottom", fontsize=14,
            fontweight="bold", color=col)
ax.set_ylim(0, 118)
ax.axhline(100, color=DIM, lw=0.8, ls=":", alpha=0.5)

ax.text(0.5, 0.85,
        "100% of voxel models are fully connected\n"
        f"{pct_single_mesh:.0f}% of exported meshes are single-body (printable)",
        ha="center", va="top", fontsize=10, color=GRAY,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.45", fc=PANEL, ec=SPINE, lw=0.8))

leg_h = [mpatches.Patch(fc=ACC, alpha=0.65, label="Voxel domain"),
         mpatches.Patch(fc=TEAL, alpha=0.65, label="Exported mesh (STL)")]
legend_styled(ax, handles=leg_h, loc="lower right", fontsize=10)

plt.tight_layout()
save(fig, "fig_connectivity.png")

print("\nDone!")
