"""
generate_model_building_figure.py
Generates a "Model Building & Deep Learning" panel figure matching the
reference slide style, using this project's hardware and architecture.

Outputs: poster_figures_v5/fig_model_building.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

# ── Color palette (matches POSTER_PLAN) ─────────────────────────────────────
BG_NAVY      = "#062B7A"
SECTION_BAR  = "#0A3D9A"
CARD_FILL    = "#F7F9FC"
CARD_BORDER  = "#B7C5E3"
TXT_DARK     = "#0B1736"
TXT_WHITE    = "#FFFFFF"
ACCENT_TEAL  = "#008C9E"
ACCENT_GOLD  = "#CFA535"
ACCENT_RED   = "#D7263D"
PILL_BG      = "#E8EEF8"

OUT_DIR = Path("poster_figures_v5")
OUT_DIR.mkdir(exist_ok=True)

# ── Figure setup ─────────────────────────────────────────────────────────────
FIG_W, FIG_H = 20, 12          # inches (poster-card proportions)
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG_NAVY)

# Transform helpers
def pct(ax, rx, ry):
    """Convert (0-1) relative coords to data coords."""
    xl, xr = ax.get_xlim(); yb, yt = ax.get_ylim()
    return xl + rx * (xr - xl), yb + ry * (yt - yb)

# ── Main axes (full figure) ───────────────────────────────────────────────────
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H)
ax.axis("off")

def card(x, y, w, h, facecolor=CARD_FILL, edgecolor=CARD_BORDER, radius=0.25, lw=1.2, zorder=2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor=facecolor, edgecolor=edgecolor,
                       linewidth=lw, zorder=zorder, clip_on=False)
    ax.add_patch(p)
    return p

def txt(x, y, s, size=11, color=TXT_DARK, bold=False, ha="left", va="center",
        zorder=5, **kw):
    w = "bold" if bold else "normal"
    ax.text(x, y, s, fontsize=size, color=color, fontweight=w,
            ha=ha, va=va, zorder=zorder, fontfamily="DejaVu Sans", **kw)

# ─────────────────────────────────────────────────────────────────────────────
# TITLE BAR
# ─────────────────────────────────────────────────────────────────────────────
TITLE_H = 1.20
card(0.15, FIG_H - TITLE_H - 0.10, FIG_W - 0.30, TITLE_H,
     facecolor=SECTION_BAR, edgecolor=SECTION_BAR, radius=0.30, lw=0)
txt(FIG_W / 2, FIG_H - 0.10 - TITLE_H / 2,
    "MODEL BUILDING & DEEP LEARNING",
    size=26, color=TXT_WHITE, bold=True, ha="center")

# Step tags (top-right corner)
for i, (lab, col) in enumerate([("PART II", ACCENT_GOLD), ("→ Methodology", "#AABBDD"), ("→ Learning", ACCENT_TEAL)]):
    bx = FIG_W - 0.20 - (2 - i) * 3.20
    by = FIG_H - 0.10 - TITLE_H / 2
    card(bx, by - 0.22, 2.90, 0.44, facecolor=col if i == 0 else "#1A3A8A",
         edgecolor=col, radius=0.20, lw=1.5, zorder=6)
    txt(bx + 1.45, by, lab, size=9, color=TXT_WHITE, ha="center", zorder=7)

# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CONTENT_Y_TOP = FIG_H - TITLE_H - 0.35
CONTENT_H     = CONTENT_Y_TOP - 0.20
MID_X         = FIG_W * 0.46          # divides left (hardware) / right (arch)
PAD           = 0.25

LEFT_X  = 0.20
LEFT_W  = MID_X - LEFT_X - PAD
RIGHT_X = MID_X + PAD
RIGHT_W = FIG_W - RIGHT_X - 0.20

# ─────────────────────────────────────────────────────────────────────────────
# LEFT PANEL — HARDWARE
# ─────────────────────────────────────────────────────────────────────────────
card(LEFT_X, 0.20, LEFT_W, CONTENT_H, radius=0.35)

hw_title_y = CONTENT_Y_TOP - 0.38
txt(LEFT_X + LEFT_W / 2, hw_title_y, "Compute Infrastructure",
    size=14, color=TXT_DARK, bold=True, ha="center")

# Divider under title
ax.plot([LEFT_X + 0.25, LEFT_X + LEFT_W - 0.25],
        [hw_title_y - 0.28, hw_title_y - 0.28],
        color=CARD_BORDER, lw=1.2, zorder=4)

# ── Hardware block helper ──────────────────────────────────────────────────
def hw_block(bx, by, bw, bh, icon, title, lines, icon_color=ACCENT_TEAL, badge=None):
    """Draw a rounded hardware sub-card."""
    card(bx, by, bw, bh, facecolor=PILL_BG, edgecolor=CARD_BORDER, radius=0.20, lw=1)
    # Icon circle
    circ = plt.Circle((bx + 0.52, by + bh / 2), 0.30, color=icon_color, zorder=5)
    ax.add_patch(circ)
    txt(bx + 0.52, by + bh / 2, icon, size=7.5, color=TXT_WHITE, ha="center", bold=True, zorder=6)
    # Title
    txt(bx + 1.05, by + bh - 0.30, title, size=12, bold=True, color=TXT_DARK)
    # Lines
    for j, line in enumerate(lines):
        txt(bx + 1.05, by + bh - 0.58 - j * 0.32, line, size=10, color=TXT_DARK)
    # Badge
    if badge:
        bpad = 0.10
        card(bx + bw - 1.55, by + bh - 0.46, 1.45, 0.34,
             facecolor=badge[1], edgecolor=badge[1], radius=0.12, lw=0, zorder=5)
        txt(bx + bw - 0.825, by + bh - 0.28, badge[0],
            size=8.5, color=TXT_WHITE, ha="center", zorder=6)

BLOCK_W  = LEFT_W - 0.50
BLOCK_X  = LEFT_X + 0.25
BLOCK_Y0 = hw_title_y - 1.05

# Block 1: Personal Laptop
hw_block(
    BLOCK_X, BLOCK_Y0 - 1.05, BLOCK_W, 1.00,
    icon="PC",
    icon_color="#3B64C8",
    title="Personal Machine (Laptop)",
    lines=["NVIDIA RTX A3000 Laptop GPU",
           "3,840 CUDA Cores  ·  6 GB VRAM"],
    badge=("Optimization", ACCENT_TEAL),
)

# Block 2: Training Cluster
hw_block(
    BLOCK_X, BLOCK_Y0 - 2.30, BLOCK_W, 1.10,
    icon="GPU",
    icon_color="#8B2FC9",
    title="Training Cluster (Remote)",
    lines=["4 × NVIDIA GB200 NVL (Blackwell)",
           "189 GB HBM3e each  ·  756 GB total",
           "16,512 CUDA Cores per GPU"],
    badge=("Ensemble Training", "#8B2FC9"),
)

# Block 3: CPU FEA
hw_block(
    BLOCK_X, BLOCK_Y0 - 3.55, BLOCK_W, 1.10,
    icon="CPU",
    icon_color=ACCENT_GOLD,
    title="FEA CPU Cluster",
    lines=["SfePy 2024 + Gmsh 4  on CPU",
           "11,178 FEA simulations",
           "~200 GB raw output  →  59 GB filtered"],
    badge=("Data Generation", ACCENT_GOLD),
)

# Speedup callout box
so_y = BLOCK_Y0 - 4.85
card(BLOCK_X, so_y, BLOCK_W, 0.75, facecolor="#0A3D9A", edgecolor=ACCENT_GOLD, radius=0.18, lw=1.8, zorder=4)
txt(BLOCK_X + BLOCK_W / 2, so_y + 0.50,
    "Surrogate replaces FEA during optimization:",
    size=10, color=TXT_WHITE, bold=False, ha="center", zorder=5)
txt(BLOCK_X + BLOCK_W / 2, so_y + 0.22,
    "median 50 s  (RTX A3000)  vs.  19–77 min  (full FEA)   →   23–92× speedup",
    size=10, color=ACCENT_GOLD, bold=True, ha="center", zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# RIGHT PANEL — MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
card(RIGHT_X, 0.20, RIGHT_W, CONTENT_H, radius=0.35)

arch_title_y = CONTENT_Y_TOP - 0.38
txt(RIGHT_X + RIGHT_W / 2, arch_title_y,
    "Starting Model Architecture  (~20 layers)",
    size=14, color=TXT_DARK, bold=True, ha="center")

ax.plot([RIGHT_X + 0.25, RIGHT_X + RIGHT_W - 0.25],
        [arch_title_y - 0.28, arch_title_y - 0.28],
        color=CARD_BORDER, lw=1.2, zorder=4)

# Architecture layers:  (label, detail, group_color)
LAYERS = [
    # group, label, detail
    ("INPUT",  "Input",            "7ch × 128³ voxel grid  +  10 geometric features"),
    ("STEM",   "Stem (Conv3d)",    "Conv3d(7→32, k=7, stride=2, pad=3)  +  BN  +  GELU"),
    ("STAGE",  "Stage 1",         "Conv3d(32→32, k=3, pad=1)  +  BN  +  GELU  +  MaxPool3d(/2)"),
    ("STAGE",  "Stage 2",         "Conv3d(32→64, k=3, pad=1)  +  BN  +  GELU  +  MaxPool3d(/2)"),
    ("STAGE",  "Stage 3",         "Conv3d(64→128, k=3, pad=1)  +  BN  +  GELU  +  MaxPool3d(/2)"),
    ("STAGE",  "Stage 4",         "Conv3d(128→256, k=3, pad=1)  +  BN  +  GELU  +  MaxPool3d(/2)"),
    ("SE",     "SE-ResBlock 1",   "BN→Conv3d(256→256)→BN→Conv3d  +  SE-Attention  +  DropPath  +  skip"),
    ("SE",     "SE-ResBlock 2",   "BN→Conv3d(256→256)→BN→Conv3d  +  SE-Attention  +  DropPath  +  skip"),
    ("SE",     "SE-ResBlock 3",   "BN→Conv3d(256→256)→BN→Conv3d  +  SE-Attention  +  DropPath  +  skip"),
    ("POOL",   "AvgPool3d",       "AdaptiveAvgPool3d(1)  →  flatten  →  256d"),
    ("POOL",   "MaxPool3d",       "AdaptiveMaxPool3d(1)  →  flatten  →  256d"),
    ("POOL",   "Concat [avg‖max]","256d + 256d  →  512d spatial embedding"),
    ("FEAT",   "Feature MLP — L1","Linear(10→128)  +  GELU  +  Dropout(0.15)"),
    ("FEAT",   "Feature MLP — L2","Linear(128→128)  +  LayerNorm  +  GELU"),
    ("FEAT",   "Concat [voxel‖geo]","512d + 128d  →  640d combined"),
    ("HEAD",   "Head — L1",       "Linear(640→512)  +  LayerNorm  +  GELU  +  Dropout(0.15)"),
    ("HEAD",   "Head — L2",       "Linear(512→256)  +  LayerNorm  +  GELU  +  Dropout(0.15)"),
    ("HEAD",   "Skip connection", "Linear(640→256)  →  add to head output  →  256d"),
    ("OUT",    "Output",          "Linear(256→3):  σ_VM  ·  δ_max  ·  compliance  C"),
    ("META",   "Loss / Opt",      "Huber loss  ·  AdamW lr=5×10⁻⁴  ·  Cosine anneal  ·  EMA 0.999"),
]

GROUP_COLORS = {
    "INPUT" : "#3B64C8",
    "STEM"  : "#2D8A6E",
    "STAGE" : ACCENT_TEAL,
    "SE"    : "#8B2FC9",
    "POOL"  : "#C47B20",
    "FEAT"  : "#1A7A48",
    "HEAD"  : ACCENT_RED,
    "OUT"   : "#0A3D9A",
    "META"  : "#555555",
}

ROW_H   = 0.42
ROW_PAD = 0.045
ARCH_Y0 = arch_title_y - 0.50      # top of first row
AX      = RIGHT_X + 0.30
AW      = RIGHT_W - 0.55
TAG_W   = 1.30
LBL_W   = 2.30
DTL_X   = AX + TAG_W + LBL_W + 0.10

for i, (grp, lbl, det) in enumerate(LAYERS):
    ry = ARCH_Y0 - i * (ROW_H + ROW_PAD) - ROW_H
    gc = GROUP_COLORS.get(grp, TXT_DARK)

    # Row background
    card(AX, ry, AW, ROW_H,
         facecolor="#FFFFFF" if i % 2 == 0 else PILL_BG,
         edgecolor=CARD_BORDER, radius=0.10, lw=0.8, zorder=3)

    # Group tag pill
    card(AX + 0.06, ry + ROW_H/2 - 0.13, TAG_W - 0.12, 0.26,
         facecolor=gc, edgecolor=gc, radius=0.10, lw=0, zorder=4)
    txt(AX + TAG_W / 2, ry + ROW_H / 2, grp,
        size=7.5, color=TXT_WHITE, ha="center", bold=True, zorder=5)

    # Layer number
    txt(AX + TAG_W + 0.12, ry + ROW_H / 2,
        f"{i+1:02d}.",
        size=9, color="#888888", ha="left", zorder=5)

    # Label (bold)
    txt(AX + TAG_W + 0.40, ry + ROW_H / 2,
        lbl, size=10, bold=True, color=TXT_DARK, ha="left", zorder=5)

    # Detail
    txt(AX + TAG_W + LBL_W + 0.08, ry + ROW_H / 2,
        det, size=9, color="#3A4A6A", ha="left", zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# Parameter count callout
# ─────────────────────────────────────────────────────────────────────────────
pc_y = ARCH_Y0 - len(LAYERS) * (ROW_H + ROW_PAD) - 0.25
card(RIGHT_X + 0.25, pc_y, RIGHT_W - 0.50, 0.62,
     facecolor="#0A3D9A", edgecolor=ACCENT_GOLD, radius=0.18, lw=1.8, zorder=4)
txt(RIGHT_X + RIGHT_W / 2, pc_y + 0.44,
    "Deep Ensemble: 5 × Surrogate3DResNet (128³)   →   8.76M params per member   →   43.8M params total",
    size=10, color=TXT_WHITE, bold=False, ha="center", zorder=5)
txt(RIGHT_X + RIGHT_W / 2, pc_y + 0.18,
    "Augmentation: random 90° rotations · horizontal flips · Gaussian noise σ=0.02 · channel dropout 10%",
    size=10, color=ACCENT_GOLD, bold=False, ha="center", zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
out_path = OUT_DIR / "fig_model_building.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight",
            facecolor=BG_NAVY, edgecolor="none")
plt.close(fig)
print(f"Saved → {out_path}")
