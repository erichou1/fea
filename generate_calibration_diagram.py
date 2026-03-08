"""
Generate a clean calibration-flow diagram for the Visual Abstract.
Inspired by the reference image but simplified and SASTO-themed.

Layout (left to right):
  [Surrogate Predictions]  ──►  [Deviation Check]  ──►  [Neural Bound]  ──►  [Validated Design]
      μ_C, σ_C                   Ĉ - C_FEA                μ + kσ ≤ 1.15         Accept / Reject
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ── Palette ─────────────────────────────────────────────────────────────────
NAVY   = "#062B7A"
BLUE   = "#0A3D9A"
LBLUE  = "#C5D4F5"
TEAL   = "#008C9E"
GOLD   = "#CFA535"
RED    = "#D7263D"
DARK   = "#0B1736"
CARD   = "#F7F9FC"
EQ_BG  = "#E8EEF2"
WHITE  = "#FFFFFF"

OUT = "poster_images_extracted/icon_calibration.png"

fig, ax = plt.subplots(figsize=(10.5, 3.6), facecolor=CARD)
ax.set_facecolor(CARD)
ax.set_xlim(0, 10.5)
ax.set_ylim(0, 3.6)
ax.axis("off")

# ── Helper: draw a rounded box ───────────────────────────────────────────────
def box(cx, cy, w, h, facecolor, edgecolor, lw=1.4, radius=0.18):
    b = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle=f"round,pad={radius}",
                       facecolor=facecolor, edgecolor=edgecolor,
                       linewidth=lw, zorder=3)
    ax.add_patch(b)

def arrow(x0, x1, y, color=TEAL, lw=2.0):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=14),
                zorder=4)

def txt(x, y, s, size=9, color=DARK, weight="normal", ha="center", va="center", style="normal"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color,
            fontweight=weight, fontstyle=style, zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN CENTRES
# ─────────────────────────────────────────────────────────────────────────────
cx = [1.1, 3.3, 5.5, 7.7, 9.7]
cy = 1.85   # vertical centre of diagram
bw, bh = 1.75, 2.20

# ── Box 1: Surrogate Predictions ─────────────────────────────────────────────
box(cx[0], cy, bw, bh, facecolor=LBLUE, edgecolor=BLUE)
txt(cx[0], cy + 0.82, "Surrogate", 9.5, BLUE, "bold")
txt(cx[0], cy + 0.52, "Predictions", 9.5, BLUE, "bold")
# small "screen" showing μ, σ bars
for i, (label, val, col) in enumerate([("μ_C", 0.72, BLUE), ("σ_C", 0.28, TEAL)]):
    by = cy - 0.05 + (0.5 - i) * 0.50
    bar = FancyBboxPatch((cx[0]-0.65, by-0.10), val * 1.3, 0.20,
                         boxstyle="round,pad=0.03", facecolor=col,
                         edgecolor="none", zorder=4)
    ax.add_patch(bar)
    txt(cx[0] - 0.68, by, label, 7.5, DARK, ha="right")
txt(cx[0], cy - 0.85, "5-member ensemble", 7.5, DARK, style="italic")

# ── Arrow 1 ──────────────────────────────────────────────────────────────────
arrow(cx[0]+bw/2+0.05, cx[1]-bw/2-0.05, cy)

# ── Box 2: Deviation (a_t - b_t style) ───────────────────────────────────────
box(cx[1], cy, bw, bh, facecolor="#FFF4E6", edgecolor=GOLD)
txt(cx[1], cy + 0.82, "Deviation:", 9.5, DARK, "bold")
# equation: Ĉ - C_FEA
ax.text(cx[1], cy + 0.40, r"$\hat{C} - C_{\mathrm{FEA}}$",
        ha="center", va="center", fontsize=11, color=DARK,
        fontweight="bold", zorder=5)
# small scatter: surrogate vs FEA
xs = np.linspace(0.3, 1.1, 12)
ys = xs + np.random.default_rng(0).normal(0, 0.06, 12)
ax.scatter(xs * 0.6 + cx[1] - 0.42,
           ys * 0.55 + cy - 1.08,
           s=12, color=TEAL, zorder=5, alpha=0.85)
ax.plot([cx[1]-0.42, cx[1]+0.37], [cy-1.08, cy-0.5],
        color=RED, lw=1.0, ls="--", zorder=4)
txt(cx[1], cy - 0.90, "surrogate vs. FEA", 7.5, DARK, style="italic")

# ── Arrow 2 ──────────────────────────────────────────────────────────────────
arrow(cx[1]+bw/2+0.05, cx[2]-bw/2-0.05, cy)

# ── Box 3: Conservative Bound ────────────────────────────────────────────────
box(cx[2], cy, bw, bh, facecolor=EQ_BG, edgecolor=BLUE)
txt(cx[2], cy + 0.82, "Conservative", 9.5, BLUE, "bold")
txt(cx[2], cy + 0.52, "Bound", 9.5, BLUE, "bold")
ax.text(cx[2], cy + 0.10, r"$\hat{y}^+ = \mu + k\sigma$",
        ha="center", va="center", fontsize=11.5, color=DARK,
        fontweight="bold", zorder=5)
ax.text(cx[2], cy - 0.32, r"$k = 1.0$  (operating pt.)",
        ha="center", va="center", fontsize=8.5, color=DARK,
        fontstyle="italic", zorder=5)
# confidence band sketch
bx_l, bx_r = cx[2]-0.62, cx[2]+0.52
for dy, col, al in [(0.12, BLUE, 0.18), (0, BLUE, 0.85), (-0.12, BLUE, 0.18)]:
    ax.plot([bx_l, bx_r], [cy-0.72+dy, cy-0.62+dy],
            color=col, lw=1.5 if dy == 0 else 0.8, alpha=al, zorder=4)
ax.fill_between([bx_l, bx_r],
                [cy-0.84, cy-0.74], [cy-0.60, cy-0.50],
                color=LBLUE, alpha=0.50, zorder=3)
txt(cx[2], cy - 0.92, "μ ± 1σ confidence band", 7.2, DARK, style="italic")

# ── Arrow 3 ──────────────────────────────────────────────────────────────────
arrow(cx[2]+bw/2+0.05, cx[3]-bw/2-0.05, cy)

# ── Box 4: Gate check ─────────────────────────────────────────────────────────
box(cx[3], cy, bw, bh, facecolor="#F0FFF4", edgecolor=TEAL)
txt(cx[3], cy + 0.82, "Constraint", 9.5, TEAL, "bold")
txt(cx[3], cy + 0.52, "Gate", 9.5, TEAL, "bold")
ax.text(cx[3], cy + 0.13,
        r"$\hat{C}^+/C_0 \leq 1.15$?",
        ha="center", va="center", fontsize=10.5, color=DARK,
        fontweight="bold", zorder=5)
# diamond decision shape
dx, dy_d = cx[3], cy - 0.55
diamond = plt.Polygon([[dx, dy_d+0.22], [dx+0.38, dy_d],
                        [dx, dy_d-0.22], [dx-0.38, dy_d]],
                       closed=True, facecolor=GOLD, edgecolor=DARK,
                       linewidth=0.8, zorder=4)
ax.add_patch(diamond)
txt(dx, dy_d, "?", 10, DARK, "bold")

# ── Arrow 4 ──────────────────────────────────────────────────────────────────
arrow(cx[3]+bw/2+0.05, cx[4]-bw/2-0.05, cy)

# ── Box 5: Outputs (accept / reject) ─────────────────────────────────────────
box(cx[4], cy+0.45, 1.55, 0.80, facecolor=TEAL, edgecolor=TEAL)
txt(cx[4], cy+0.45, "✓  Accept", 9.5, WHITE, "bold")

box(cx[4], cy-0.45, 1.55, 0.80, facecolor=RED, edgecolor=RED)
txt(cx[4], cy-0.45, "✗  Reject", 9.5, WHITE, "bold")

txt(cx[4], cy-1.02, "commit / undo + halve B", 7.2, DARK, style="italic")

# ── Section title bar (top) ───────────────────────────────────────────────────
title_bar = FancyBboxPatch((0.05, 3.18), 10.40, 0.34,
                            boxstyle="round,pad=0.04",
                            facecolor=BLUE, edgecolor="none", zorder=2)
ax.add_patch(title_bar)
txt(5.25, 3.35, "FEA Calibration  ·  Surrogate Prediction  →  Conservative Bound  →  Gate Check  →  Design Decision",
    8.5, WHITE, "bold")

# ── Thin gold rule under title bar ───────────────────────────────────────────
ax.plot([0.05, 10.45], [3.16, 3.16], color=GOLD, lw=1.5, zorder=3)

plt.tight_layout(pad=0.1)
plt.savefig(OUT, dpi=280, bbox_inches="tight", facecolor=CARD)
plt.close()
print(f"Saved → {OUT}")
