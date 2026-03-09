"""
Surrogate Learning Pipeline — visual-only transparent PNGs.

  fig_sur1_voxels.png     — 3-D scatter of building voxels coloured by part
  fig_sur2_channels.png   — 7-channel input slices (imshow panels)
  fig_sur3_arch.png       — isometric 3-D feature-map block diagram
  fig_sur4_ensemble.png   — deep-ensemble flow: 5 CNN towers → 3 outputs
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from pathlib import Path
import json, colorsys

# ── palette ─────────────────────────────────────────────────────────────────
BG   = "#0d0d0d"
GRAY = "#d8d8d8"
ACC  = "#4fc3f7"
DIM  = "#666666"

PART_HEX   = {1: "#3a8fd1", 2: "#d4700a", 3: "#29913a", 4: "#9a7720"}
PART_NAMES = {1: "Exterior Wall", 2: "Interior Wall", 3: "Roof", 4: "Floor"}

# ── data ────────────────────────────────────────────────────────────────────
SAMPLE = Path("fea_ml/data/runs_real_128/00000")
OUT    = Path("figures")
OUT.mkdir(exist_ok=True)

occ  = np.load(SAMPLE / "occ.npz")["data"].astype(bool)
part = np.load(SAMPLE / "part.npz")["data"].astype(int)


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=180, bbox_inches="tight",
                facecolor="none", transparent=True)
    plt.close(fig)
    print(f"  saved → {path}")


# ── iso-face colour helpers ──────────────────────────────────────────────────
def _hex_hls(h):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
    return colorsys.rgb_to_hls(r, g, b)

def _from_hls(hls, alpha):
    return (*colorsys.hls_to_rgb(*hls), alpha)

def face_f(h):   # front – slightly brightened
    hh, ll, ss = _hex_hls(h); return _from_hls((hh, min(1, ll*1.15), ss), 0.92)
def face_t(h):   # top – medium
    hh, ll, ss = _hex_hls(h); return _from_hls((hh, max(0, ll*0.80), ss), 0.85)
def face_r(h):   # right – darkened
    hh, ll, ss = _hex_hls(h); return _from_hls((hh, max(0, ll*0.52), ss), 0.80)


# ═══════════════════════════════════════════════════════════════════════════
# FIG 1 — 3-D voxel scatter, coloured by structural part
# ═══════════════════════════════════════════════════════════════════════════
def fig_voxels():
    rng = np.random.default_rng(42)

    fig = plt.figure(figsize=(8, 7), facecolor="none")
    ax  = fig.add_subplot(111, projection="3d", facecolor=BG)

    for pid, col in PART_HEX.items():
        mask = occ & (part == pid)
        xs, ys, zs = np.where(mask)
        if xs.size == 0:
            continue
        n   = min(4000, xs.size)
        idx = rng.choice(xs.size, n, replace=False)
        ax.scatter(xs[idx], ys[idx], zs[idx],
                   c=col, s=3.5, alpha=0.72,
                   edgecolors="none", depthshade=True,
                   label=PART_NAMES[pid])

    ax.set_facecolor(BG)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#222")
    ax.grid(False)
    ax.set_axis_off()
    ax.view_init(elev=22, azim=-50)

    handles = [
        mpatches.Patch(facecolor=c, edgecolor="none", label=PART_NAMES[pid])
        for pid, c in PART_HEX.items()
    ]
    ax.legend(handles=handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.04),
              ncol=4, frameon=False,
              labelcolor=GRAY, fontsize=10,
              handletextpad=0.4, columnspacing=1.2)

    ax.set_title("128³ Voxel Grid  —  Input Geometry  (coloured by part)",
                 color=ACC, fontsize=12, fontweight="bold", pad=6)
    save(fig, "fig_sur1_voxels.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 2 — 7-channel input tensor panels (imshow slices)
# ═══════════════════════════════════════════════════════════════════════════
def fig_channels():
    """
    6 active input channels in a 3+3 layout (ch0–ch5).
    ch 0 = occupancy
    ch 1..5 = binary masks for part == 0..4
    (ch 6 / part==5 has no data in this dataset and is omitted)
    """
    import matplotlib.gridspec as gridspec

    # channel definitions: (label, mask_fn, best_z_hint, cmap, border_hex)
    CHAN_DEFS = [
        ("Occupancy",      lambda z: occ[:, :, z].astype(float),         51, plt.cm.Blues,   ACC),
        ("Air",            lambda z: (part[:, :, z] == 0).astype(float), 51, plt.cm.Purples, "#9a70cc"),
        ("Exterior Wall",  lambda z: (part[:, :, z] == 1).astype(float), 51, plt.cm.Oranges, PART_HEX[1]),
        ("Interior Wall",  lambda z: (part[:, :, z] == 2).astype(float), 50, plt.cm.Oranges, PART_HEX[2]),
        ("Roof",           lambda z: (part[:, :, z] == 3).astype(float), 66, plt.cm.Greens,  PART_HEX[3]),
        ("Floor",          lambda z: (part[:, :, z] == 4).astype(float), 46, plt.cm.YlOrBr,  PART_HEX[4]),
    ]

    # Pick the z-slice with the most active voxels per channel
    slices, borders, labels, cmaps = [], [], [], []
    for lbl, fn, hint, cmap, bord in CHAN_DEFS:
        counts = np.array([fn(z).sum() for z in range(occ.shape[2])])
        best_z = int(np.argmax(counts)) if counts.max() > 0 else hint
        slices.append(fn(best_z))
        labels.append(lbl)
        cmaps.append(cmap)
        borders.append(bord)

    tags = [f"ch {i}" for i in range(6)]

    fig = plt.figure(figsize=(10, 6.8), facecolor="none")
    fig.patch.set_facecolor("none")

    # 6-column grid, 2 equal rows of 3
    gs = gridspec.GridSpec(2, 6, figure=fig,
                           hspace=0.10, wspace=0.06,
                           left=0.01, right=0.99,
                           top=0.89, bottom=0.01)
    positions = [
        gs[0, 0:2], gs[0, 2:4], gs[0, 4:6],
        gs[1, 0:2], gs[1, 2:4], gs[1, 4:6],
    ]

    for sl, cmap, lbl, bord, tag, pos in zip(slices, cmaps, labels, borders, tags, positions):
        ax = fig.add_subplot(pos)
        ax.set_facecolor(BG)
        ax.imshow(sl.T, origin="lower", cmap=cmap,
                  interpolation="nearest", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(bord)
            spine.set_linewidth(3.5)
        ax.text(0.03, 0.97, tag, transform=ax.transAxes,
                ha="left", va="top", color=bord,
                fontsize=14, fontweight="bold",
                bbox=dict(fc="#000000", ec="none", alpha=0.70, pad=4))
        ax.text(0.50, 0.04, lbl, transform=ax.transAxes,
                ha="center", va="bottom", color="white",
                fontsize=12, fontweight="bold",
                bbox=dict(fc="#000000", ec="none", alpha=0.70, pad=4))

    fig.suptitle("Input Tensor  ·  6 active channels (ch0–ch5)  ·  128 × 128 slice",
                 color=ACC, fontsize=13, fontweight="bold")
    save(fig, "fig_sur2_channels.png")


# ═══════════════════════════════════════════════════════════════════════════
# Feature Vector — 10 bright chips, zero overlap
# ═══════════════════════════════════════════════════════════════════════════
def fig_feature_vector():
    """
    Poster-panel style (reference image).
    Shows WHAT the feature vector is: 10 per-simulation scalars
    describing material & loading, concatenated with the voxel CNN output.
    """
    FEATURES = [
        # (symbol, full_name, value, group)
        ("E",      "Young's Modulus",   "25 GPa",        "mat"),
        ("\u03c1",      "Density",           "2400 kg/m\u00b3",  "mat"),
        ("\u03c3_y",    "Yield Stress",      "30 MPa",        "mat"),
        ("\u03bd",      "Poisson's Ratio",   "0.20",          "mat"),
        ("g",      "Gravity",           "9.81 m/s\u00b2",   "load"),
        ("w",      "Wind Pressure",     "0.60 kPa",      "load"),
        ("gravity","Load: Gravity",     "\u25a0  ON",          "flag"),
        ("wind",   "Load: Wind",        "\u25a0  ON",          "flag"),
        ("seismic","Load: Seismic",     "\u25a1  off",         "flag"),
        ("vox \u0394", "Voxel Size",        "1.62 cm",       "geom"),
    ]
    GROUP_COL  = {"mat": "#1a5fa8", "load": "#1a8a50", "flag": "#c07010", "geom": "#7040b0"}
    GROUP_NAME = {"mat": "Material", "load": "Loading", "flag": "Load Case", "geom": "Geometry"}
    GROUP_ORDER = ["mat", "load", "flag", "geom"]

    # Layout constants — computed bottom-up so nothing clips
    CW, CH = 1.72, 1.68     # chip size
    HGAP, VGAP = 0.26, 0.44
    NCOLS = 5
    NROWS = 2
    HEADER_H = 0.38         # group header strip height
    LEGEND_H  = 0.55        # bottom legend area
    TOP_PAD   = 0.20        # padding above chips

    # Total data height (2 rows of chips + 1 gap + header + legend)
    CHIPS_H   = NROWS * CH + (NROWS - 1) * VGAP
    TOTAL_H   = LEGEND_H + CHIPS_H + HEADER_H + TOP_PAD
    TOTAL_W   = NCOLS * CW + (NCOLS - 1) * HGAP

    # y-baseline: chips start at LEGEND_H
    # row 1 (bottom): y = LEGEND_H
    # row 0 (top):    y = LEGEND_H + CH + VGAP
    def chip_y(row_idx):   # row_idx=0 is TOP row
        return LEGEND_H + (NROWS - 1 - row_idx) * (CH + VGAP)

    fig, ax = plt.subplots(figsize=(TOTAL_W * 1.62, TOTAL_H * 1.62), facecolor="none")
    # Light misty panel background
    ax.set_facecolor("#141c28")
    ax.set_axis_off()
    ax.set_xlim(-0.20, TOTAL_W + 0.20)
    ax.set_ylim(-0.10, TOTAL_H + 0.10)

    # ── panel outline ──────────────────────────────────────
    ax.add_patch(FancyBboxPatch((-0.15, -0.06), TOTAL_W + 0.30, TOTAL_H + 0.12,
                                boxstyle="round,pad=0.10",
                                fc="#1a2434", ec="#3a6fa8", lw=2.5, zorder=0))

    # ── title banner ──────────────────────────────────────
    banner_y = LEGEND_H + CHIPS_H + VGAP * 0.15
    ax.add_patch(FancyBboxPatch((-0.15, banner_y), TOTAL_W + 0.30, HEADER_H + TOP_PAD,
                                boxstyle="round,pad=0.04",
                                fc="#1e5fa8", ec="none", zorder=1))
    ax.text(TOTAL_W / 2, banner_y + (HEADER_H + TOP_PAD) / 2,
            "Feature Vector  ·  10 Scalars Describing the Simulation",
            ha="center", va="center", color="white",
            fontsize=14, fontweight="bold", zorder=3)

    # ── group header bars (one per group, spanning their columns) ────────
    groups_x = {}   # group -> list of column x0s
    for i, (_, _, _, grp) in enumerate(FEATURES):
        ci = i % NCOLS
        x0 = ci * (CW + HGAP)
        groups_x.setdefault(grp, []).append(x0)

    for grp in GROUP_ORDER:
        xs = groups_x.get(grp, [])
        if not xs:
            continue
        col = GROUP_COL[grp]
        gx0 = min(xs) - 0.06
        gx1 = max(xs) + CW + 0.06
        gy  = chip_y(0) + CH + 0.06
        ax.add_patch(FancyBboxPatch((gx0, gy), gx1 - gx0, HEADER_H * 0.70,
                                    boxstyle="round,pad=0.05",
                                    fc=col, ec="none", alpha=0.85, zorder=2))
        ax.text((gx0 + gx1) / 2, gy + HEADER_H * 0.35,
                GROUP_NAME[grp], ha="center", va="center",
                color="white", fontsize=11, fontweight="bold", zorder=4)

    # ── chips ────────────────────────────────────────────
    for i, (sym, name, val, grp) in enumerate(FEATURES):
        ci = i % NCOLS
        ri = i // NCOLS
        x0 = ci * (CW + HGAP)
        y0 = chip_y(ri)
        col = GROUP_COL[grp]

        ax.add_patch(FancyBboxPatch(
            (x0, y0), CW, CH,
            boxstyle="round,pad=0.10",
            fc=col, ec="white", lw=2.2, alpha=0.95, zorder=2))

        # index
        ax.text(x0 + 0.10, y0 + CH - 0.08, f"[{i}]",
                ha="left", va="top", color="white",
                fontsize=9, alpha=0.70, zorder=4)

        # symbol (large, centre-upper)
        ax.text(x0 + CW / 2, y0 + CH * 0.65, sym,
                ha="center", va="center", color="white",
                fontsize=20, fontweight="bold", zorder=4)

        # full name (small, middle)
        ax.text(x0 + CW / 2, y0 + CH * 0.38, name,
                ha="center", va="center", color="#cce4ff",
                fontsize=8.5, zorder=4)

        # value (bottom, monospace)
        ax.text(x0 + CW / 2, y0 + CH * 0.16, val,
                ha="center", va="center", color="white",
                fontsize=10, fontweight="bold",
                fontfamily="monospace", zorder=4)

    # ── bottom legend ──────────────────────────────────────
    leg_items = [(GROUP_COL[g], GROUP_NAME[g] + " props") for g in GROUP_ORDER]
    total_leg_w = len(leg_items) * 2.30
    lx = (TOTAL_W - total_leg_w) / 2
    ly = LEGEND_H / 2 - 0.12
    for col, lbl in leg_items:
        ax.add_patch(FancyBboxPatch((lx, ly - 0.14), 0.32, 0.28,
                                    boxstyle="round,pad=0.04",
                                    fc=col, ec="none", zorder=3))
        ax.text(lx + 0.44, ly - 0.01, lbl,
                ha="left", va="center", color=GRAY,
                fontsize=10, zorder=4)
        lx += 2.30

    # purpose note
    ax.text(TOTAL_W / 2, 0.06,
            "Concatenated with pooled 3D-CNN embedding before the prediction head",
            ha="center", va="bottom", color=DIM, fontsize=9)

    save(fig, "fig_sur2_feature_vector.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 3 — Isometric 3-D feature-map blocks (architecture)
# ═══════════════════════════════════════════════════════════════════════════
def fig_arch():
    """
    Reference-style pipeline diagram.
    3 clearly labelled zones with a bold banner each:
      ZONE 1: Voxel Input (iso cube)  +  Feature Vector (chip stack)
      ZONE 2: 3D ResNet (shrinking cube cascade)
      ZONE 3: Structural Predictions (3 coloured output nodes)
    Thick black arrows connect zones.
    """
    FW, FH = 16.0, 6.8
    fig, ax = plt.subplots(figsize=(FW, FH), facecolor="none")
    ax.set_facecolor("#111820")
    ax.set_axis_off()
    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)

    # ───────────────────────────────────────────────────────────────────────────
    # Helper: draw an isometric cube
    # ───────────────────────────────────────────────────────────────────────────
    def iso_cube(cx, cy, sz, depth, col, zorder=3):
        """Draw iso cube centred on (cx,cy) with face size sz, depth depth."""
        ix = depth * 0.50
        iy = depth * 0.30
        x0 = cx - sz / 2
        y0 = cy - sz / 2
        # front
        ax.add_patch(Polygon([
            [x0, y0],[x0+sz, y0],[x0+sz, y0+sz],[x0, y0+sz]],
            fc=face_f(col), ec=col, lw=1.8, zorder=zorder, closed=True))
        # top
        ax.add_patch(Polygon([
            [x0, y0+sz],[x0+sz, y0+sz],
            [x0+sz+ix, y0+sz+iy],[x0+ix, y0+sz+iy]],
            fc=face_t(col), ec=col, lw=1.2, zorder=zorder, closed=True))
        # right
        ax.add_patch(Polygon([
            [x0+sz, y0],[x0+sz+ix, y0+iy],
            [x0+sz+ix, y0+sz+iy],[x0+sz, y0+sz]],
            fc=face_r(col), ec=col, lw=1.2, zorder=zorder, closed=True))
        return x0+sz+ix   # right tip x

    # ───────────────────────────────────────────────────────────────────────────
    # Zone backgrounds + header banners
    # ───────────────────────────────────────────────────────────────────────────
    ZONES = [
        (0.20,  3.50, "#1a4070", "ZONE 1",  "\u25b6  Input",       "Voxel Grid + Feature Vector"),
        (3.80,  8.00, "#1a5a3a", "ZONE 2",  "\u25b6  3D ResNet",   "Spatial compression + feature extraction"),
        (12.10, 3.90, "#6a3010", "ZONE 3",  "\u25b6  Predictions", "Structural performance outputs"),
    ]
    for zx, zw, zcol, ztag, zhead, zsub in ZONES:
        # zone panel
        ax.add_patch(FancyBboxPatch(
            (zx, 0.4), zw, FH - 1.0,
            boxstyle="round,pad=0.15",
            fc=zcol, ec="none", alpha=0.18, zorder=1))
        # header banner
        ax.add_patch(FancyBboxPatch(
            (zx, FH - 1.05), zw, 0.72,
            boxstyle="round,pad=0.10",
            fc=zcol, ec="none", alpha=0.90, zorder=2))
        ax.text(zx + zw / 2, FH - 0.70, zhead,
                ha="center", va="center", color="white",
                fontsize=13, fontweight="bold", zorder=5)
        ax.text(zx + zw / 2, FH - 1.22, zsub,
                ha="center", va="top", color="#aaaaaa",
                fontsize=9, zorder=5)

    # ───────────────────────────────────────────────────────────────────────────
    # ZONE 1: input voxel cube + feature vector mini-chips
    # ───────────────────────────────────────────────────────────────────────────
    iso_cube(1.45, 3.55, 1.80, 0.55, "#2a6aaa", zorder=4)
    ax.text(1.45, 3.55 + 0.02, "7 ch",
            ha="center", va="center", color="white",
            fontsize=13, fontweight="bold", zorder=6)
    ax.text(1.45, 3.55 - 0.36, "128³",
            ha="center", va="center", color=ACC,
            fontsize=11, fontfamily="monospace", zorder=6)
    ax.text(1.45, 1.70, "Voxel Grid",
            ha="center", va="top", color=GRAY,
            fontsize=11, fontweight="bold")

    # mini feature-vector chips (2 cols × 5 rows)
    FV_LABELS = [("E","\u03c1","\u03c3_y","\u03bd","g"),("w","grav","wind","seis","vox")]
    FV_COLS   = ["#1a5fa8","#1a5fa8","#1a5fa8","#1a5fa8","#1a8a50",
                 "#1a8a50","#c07010","#c07010","#c07010","#7040b0"]
    MCW, MCH  = 0.52, 0.44
    MGAP      = 0.06
    mx0       = 2.38
    my_base   = 1.72
    for row in range(5):
        for col in range(2):
            idx  = col * 5 + row
            sym  = FV_LABELS[col][row]
            fc   = FV_COLS[idx]
            mx   = mx0 + col * (MCW + MGAP * 2.5)
            my   = my_base + row * (MCH + MGAP)
            ax.add_patch(FancyBboxPatch(
                (mx, my), MCW, MCH,
                boxstyle="round,pad=0.05",
                fc=fc, ec="white", lw=1.2, zorder=4))
            ax.text(mx + MCW / 2, my + MCH / 2, sym,
                    ha="center", va="center", color="white",
                    fontsize=9.5, fontweight="bold", zorder=5)
    ax.text(mx0 + (MCW + MGAP * 2.5 + MCW) / 2 - MGAP, my_base - 0.24,
            "10-dim features",
            ha="center", va="top", color=GRAY,
            fontsize=10, fontweight="bold")

    # ───────────────────────────────────────────────────────────────────────────
    # ZONE 1 -> ZONE 2: thick arrow
    # ───────────────────────────────────────────────────────────────────────────
    ax.annotate("", xy=(3.78, 3.40), xytext=(3.20, 3.40),
                arrowprops=dict(arrowstyle="-|>",
                                color="white", lw=3.0, mutation_scale=22))

    # ───────────────────────────────────────────────────────────────────────────
    # ZONE 2: cascading shrinking cubes  (spatial 128→/4, depth grows)
    # ───────────────────────────────────────────────────────────────────────────
    # (label, face_sz, depth, col, cx, cy)
    CNN_CUBES = [
        ("Input\n128³",  1.30, 0.25, "#2a6aaa", 4.55, 3.60),
        ("Stem\n32³",   0.90, 0.38, "#2a8a7a", 5.80, 3.42),
        ("St. 1\n32³",  0.90, 0.46, "#2a9a5a", 6.98, 3.42),
        ("St. 2\n16³",  0.68, 0.54, "#3a9a3a", 8.06, 3.34),
        ("St. 3\n8³",   0.50, 0.62, "#7a9a1a", 9.00, 3.25),
        ("St. 4\n4³",   0.36, 0.70, "#9a6a10", 9.82, 3.18),
        ("Pool",         0.20, 0.78, "#9a4820", 10.55, 3.10),
    ]
    for label, sz, dep, col, cx, cy in CNN_CUBES:
        rx = iso_cube(cx, cy, sz, dep, col, zorder=4)
        # label below cube
        ax.text(cx, cy - sz / 2 - 0.20, label,
                ha="center", va="top", color=GRAY,
                fontsize=8.5, linespacing=1.3)

    # connecting arrows between cubes
    CUB_MID_Y = 3.40
    CARROW_XS = [(5.22, 5.56), (6.40, 6.74), (7.46, 7.80),
                 (8.40, 8.72), (9.22, 9.58), (9.96, 10.30)]
    for x1, x2 in CARROW_XS:
        ax.annotate("", xy=(x2, CUB_MID_Y), xytext=(x1, CUB_MID_Y),
                    arrowprops=dict(arrowstyle="-|>", color="#4488aa",
                                    lw=1.8, mutation_scale=12), zorder=6)

    # feature MLP inject arrow (from below into pool stage)
    ax.annotate("", xy=(10.55, 3.10 - 0.20 / 2),
                xytext=(10.55, 1.90),
                arrowprops=dict(arrowstyle="-|>", color="#c07010",
                                lw=1.5, mutation_scale=10), zorder=6)
    ax.text(10.55, 1.80, "+feature\nvector",
            ha="center", va="top", color="#c07010",
            fontsize=9, fontweight="bold", linespacing=1.3)

    # ───────────────────────────────────────────────────────────────────────────
    # ZONE 2 -> ZONE 3: thick arrow
    # ───────────────────────────────────────────────────────────────────────────
    ax.annotate("", xy=(12.08, 3.40), xytext=(11.50, 3.40),
                arrowprops=dict(arrowstyle="-|>",
                                color="white", lw=3.0, mutation_scale=22))

    # ──────────────────────────────────────────────────────────────════════════
    # ZONE 3: 3 prediction output nodes
    # ───────────────────────────────────────────────────────────────────────────
    OUTS = [
        (5.30, "\u03c3\u1d6a", "Von Mises\nStress",  "#ff6666"),
        (3.40, "\u03b4",   "Max\nDisplacement", "#4fc3f7"),
        (1.60, "C",   "Compliance",         "#66ee88"),
    ]
    NR = 0.52
    for oy, sym, lbl, col in OUTS:
        ax.annotate("", xy=(12.50, oy), xytext=(12.20, 3.40),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                    lw=1.8, mutation_scale=12))
        circ = plt.Circle((13.30, oy), NR, fc="#111", ec=col, lw=2.8, zorder=5)
        ax.add_patch(circ)
        ax.text(13.30, oy + 0.10, sym,
                ha="center", va="center", color=col,
                fontsize=14, fontweight="bold", zorder=6)
        ax.text(13.30, oy - 0.22, "\u03bc \u00b1 \u03c3",
                ha="center", va="center", color="#888",
                fontsize=8.5, fontfamily="monospace", zorder=6)
        ax.text(14.00, oy, lbl,
                ha="left", va="center", color=GRAY,
                fontsize=11, fontweight="bold", linespacing=1.4)

    # ── overall title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        "Surrogate Model Pipeline  ·  Predicts structural performance without running FEA",
        color=ACC, fontsize=14, fontweight="bold", pad=10)
    save(fig, "fig_sur3_arch.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4 — Deep ensemble: 5 CNN towers → aggregation → 3 outputs
# ═══════════════════════════════════════════════════════════════════════════
def fig_ensemble():
    N         = 5
    FIG_W     = 16.0
    FIG_H     = 8.0
    MEMBER_YS = np.linspace(1.0, 7.0, N)

    T_COLS  = ["#2a6a9a", "#2a8a6a", "#4a8a2a", "#8a8a1a", "#9a5010"]
    # tower layers: wide at left (input), narrow at right (deep features)
    LAYER_WS = [0.80, 0.68, 0.56, 0.44, 0.32, 0.20]
    L_H      = 0.130   # taller strips → visually prominent
    T_W      = 0.90
    T_H      = len(LAYER_WS) * L_H
    LCOLS    = ["#2a6a9a","#2a7a7a","#2a8a5a","#4a8a2a","#7a7a1a","#9a5010"]

    IN_CX, IN_CY = 1.10, 4.00
    TWR_X        = 3.30
    AGG_CX       = 10.50
    AGG_CY       = 4.00
    OUT_X        = 12.20
    OUT_YS       = [6.10, 4.00, 1.90]
    OUT_SYM      = ["σ_VM",   "δ_max",  "C"]
    OUT_LBL      = ["Von Mises\nStress",
                    "Max\nDisplacement",
                    "Compliance"]
    OUT_COL      = ["#ff6666", "#4fc3f7", "#66ee88"]
    NODE_R       = 0.52

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor="none")
    ax.set_facecolor(BG)
    ax.set_axis_off()
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)

    # ── input iso-cube ───────────────────────────────────────────────────────
    IW, IH, IISO = 1.0, 1.5, 0.34
    ix0 = IN_CX - IW / 2
    iy0 = IN_CY - IH / 2
    ax.add_patch(Polygon([
        [ix0, iy0],[ix0+IW, iy0],[ix0+IW, iy0+IH],[ix0, iy0+IH]],
        fc=face_f("#2a5a8a"), ec="#2a5a8a", lw=2.0, zorder=3))
    ax.add_patch(Polygon([
        [ix0, iy0+IH],[ix0+IW, iy0+IH],
        [ix0+IW+IISO, iy0+IH+IISO*0.6],[ix0+IISO, iy0+IH+IISO*0.6]],
        fc=face_t("#2a5a8a"), ec="#2a5a8a", lw=1.0, zorder=3))
    ax.add_patch(Polygon([
        [ix0+IW, iy0],[ix0+IW+IISO, iy0+IISO*0.6],
        [ix0+IW+IISO, iy0+IH+IISO*0.6],[ix0+IW, iy0+IH]],
        fc=face_r("#2a5a8a"), ec="#2a5a8a", lw=1.0, zorder=3))

    ax.text(IN_CX, IN_CY + 0.22, "7 ch",
            ha="center", va="center", color=GRAY,
            fontsize=13, fontweight="bold", zorder=5)
    ax.text(IN_CX, IN_CY - 0.22, "128³",
            ha="center", va="center", color=ACC,
            fontsize=12, fontfamily="monospace", zorder=5)
    ax.text(IN_CX, iy0 - 0.18, "Input Voxel\nGrid",
            ha="center", va="top", color=DIM,
            fontsize=11, linespacing=1.4)

    cube_rx = ix0 + IW + IISO

    # ── 5 CNN towers ────────────────────────────────────────────────────────
    for mi, (my, tc) in enumerate(zip(MEMBER_YS, T_COLS)):
        ty_bot = my - T_H / 2

        ax.annotate("", xy=(TWR_X, my), xytext=(cube_rx + 0.08, IN_CY),
                    arrowprops=dict(arrowstyle="-|>", color="#2a4a6a",
                                    lw=1.3, mutation_scale=10))

        for li, (lw, lc) in enumerate(zip(LAYER_WS, LCOLS)):
            lx = TWR_X + (T_W - lw) / 2
            ly = ty_bot + li * L_H
            ax.add_patch(FancyBboxPatch(
                (lx, ly), lw, L_H * 0.88,
                boxstyle="round,pad=0.012",
                fc=lc, ec=tc, lw=1.0, zorder=3))

        # member number — above the tower, colour-matched
        ax.text(TWR_X + T_W / 2, my + T_H / 2 + 0.10,
                f"M{mi+1}",
                ha="center", va="bottom",
                color=tc, fontsize=11, fontweight="bold", zorder=5)

        ax.annotate("", xy=(AGG_CX - 0.50, AGG_CY),
                    xytext=(TWR_X + T_W + 0.10, my),
                    arrowprops=dict(arrowstyle="-|>", color="#2a4a6a",
                                    lw=1.3, mutation_scale=10))

    # ── aggregation diamond ─────────────────────────────────────────────────
    AR = 0.65
    ax.add_patch(Polygon([
        [AGG_CX,           AGG_CY + AR],
        [AGG_CX + AR*0.72, AGG_CY],
        [AGG_CX,           AGG_CY - AR],
        [AGG_CX - AR*0.72, AGG_CY]],
        fc="#1e1008", ec="#ffb347", lw=2.8, zorder=4))
    ax.text(AGG_CX, AGG_CY + 0.18, "μ",
            ha="center", va="center", color="#ffb347",
            fontsize=22, fontweight="bold", zorder=5)
    ax.text(AGG_CX, AGG_CY - 0.28, "± σ",
            ha="center", va="center", color="#ffb347",
            fontsize=14, zorder=5)
    ax.text(AGG_CX, AGG_CY + AR + 0.16, "Ensemble",
            ha="center", va="bottom", color="#ffb347",
            fontsize=12, fontweight="bold")

    ax.annotate("",
        xy=(AGG_CX + AR*0.72 + 0.24, AGG_CY),
        xytext=(AGG_CX + AR*0.72, AGG_CY),
        arrowprops=dict(arrowstyle="-|>", color=ACC, lw=2.4,
                        mutation_scale=16))

    # ── 3 output nodes ──────────────────────────────────────────────────────
    for oy, sym, lbl, col in zip(OUT_YS, OUT_SYM, OUT_LBL, OUT_COL):
        ax.annotate("",
            xy=(OUT_X, oy),
            xytext=(AGG_CX + AR*0.72 + 0.24, AGG_CY),
            arrowprops=dict(arrowstyle="-|>", color=col,
                            lw=1.8, mutation_scale=11))

        circ = plt.Circle((OUT_X + NODE_R, oy), NODE_R,
                           fc="#111111", ec=col, lw=3.0, zorder=4)
        ax.add_patch(circ)
        ax.text(OUT_X + NODE_R, oy + 0.14, sym,
                ha="center", va="center", color=col,
                fontsize=13, fontweight="bold", zorder=5)
        ax.text(OUT_X + NODE_R, oy - 0.20, "μ ± σ",
                ha="center", va="center", color="#888",
                fontsize=10, fontfamily="monospace", zorder=5)
        # label to the right — guaranteed clear of circle
        ax.text(OUT_X + NODE_R * 2 + 0.28, oy, lbl,
                ha="left", va="center", color=GRAY,
                fontsize=12, linespacing=1.5, zorder=5)

    # ── footer ──────────────────────────────────────────────────────────────
    ax.text(FIG_W / 2, 0.18,
            "5 independently trained members  ·  11,178 FEA training simulations",
            ha="center", va="bottom", color=DIM, fontsize=10)

    ax.set_title("Deep Ensemble  ·  5 × 3D-ResNet  →  Mean ± Uncertainty",
                 color=ACC, fontsize=14, fontweight="bold", pad=10)
    save(fig, "fig_sur4_ensemble.png")


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("SUR 1 — 3-D voxel scatter …")
    fig_voxels()
    print("SUR 2a — channel slice panels \u2026")
    fig_channels()
    print("SUR 2b — feature vector chips \u2026")
    fig_feature_vector()
    print("SUR 3 — architecture block diagram \u2026")
    fig_arch()
    print("SUR 4 — deep ensemble flow \u2026")
    fig_ensemble()
    print("\nDone — all figures saved to figures/")
