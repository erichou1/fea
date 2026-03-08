"""
Generate a single calibration icon for the Visual Abstract.

Icon: confidence-interval bar chart — prediction bounds vs. threshold line.
Per-design error bars (μ + kσ), gold threshold, teal accept / red reject colouring.
Matches the 5 diagram-icon style (3.2 × 2.6 in, CARD background, set_title label).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

# ── Palette ─────────────────────────────────────────────────────────────────
NAVY  = "#062B7A"
BLUE  = "#0A3D9A"
LBLUE = "#C5D4F5"
TEAL  = "#008C9E"
GOLD  = "#CFA535"
RED   = "#D7263D"
DARK  = "#0B1736"
CARD  = "#F7F9FC"
WHITE = "#FFFFFF"

OUT = "poster_images_extracted/icon_calibration.png"

W, H = 3.2, 2.6
DPI  = 220

rng = np.random.default_rng(42)

# ── Fake per-design data (6 designs) ─────────────────────────────────────────
n       = 6
# "true" compliance ratios spread around 1.0
true_c  = np.array([0.75, 0.88, 0.96, 1.05, 1.10, 1.20])
# surrogate mean (slightly under-predicts true)
mu      = true_c * rng.uniform(0.90, 0.98, n)
# surrogate std
sigma   = rng.uniform(0.04, 0.09, n)
k       = 1.0                     # operating k
bound   = mu + k * sigma          # conservative upper bound
thresh  = 1.15                    # acceptance threshold

xs = np.arange(n)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(W, H), facecolor=CARD)
ax.set_facecolor(CARD)

# ── Bars: μ as dot, error bar showing [μ, μ+kσ] ──────────────────────────────
for i in range(n):
    accept = bound[i] <= thresh
    bar_col  = TEAL  if accept else RED
    edge_col = TEAL  if accept else RED

    # vertical bar from μ up to bound
    ax.plot([xs[i], xs[i]], [mu[i], bound[i]],
            color=bar_col, lw=3.5, solid_capstyle="round", zorder=3)
    # cap at top (bound)
    ax.plot([xs[i]-0.18, xs[i]+0.18], [bound[i], bound[i]],
            color=bar_col, lw=2.2, zorder=4)
    # dot at μ
    ax.plot(xs[i], mu[i], 'o', color=bar_col, ms=6, zorder=5,
            markeredgecolor=DARK, markeredgewidth=0.5)

# ── Gold threshold line ───────────────────────────────────────────────────────
ax.axhline(thresh, color=GOLD, lw=2.0, ls="--", zorder=6)
ax.text(n - 0.1, thresh + 0.022,
        r"$\hat{C}^+/C_0 \leq 1.15$",
        ha="right", va="bottom", fontsize=7.5, color=GOLD,
        fontweight="bold", zorder=7)

# ── Legend dots ───────────────────────────────────────────────────────────────
ax.plot([], [], 'o-', color=TEAL, lw=2, label="Accept  ✓", ms=5)
ax.plot([], [], 'o-', color=RED,  lw=2, label="Reject  ✗",  ms=5)
ax.legend(fontsize=7, loc="upper left", frameon=False,
          labelcolor=DARK, handlelength=1.2)

# ── Axes cosmetics ────────────────────────────────────────────────────────────
ax.set_xlim(-0.55, n - 0.45)
ax.set_ylim(0.58, 1.38)
ax.set_xticks(xs)
ax.set_xticklabels([f"D{i+1}" for i in range(n)],
                   fontsize=7.5, color=DARK)
ax.set_ylabel(r"Compliance ratio  $\hat{C}^+/C_0$",
              fontsize=7.5, color=DARK)
ax.tick_params(axis="y", labelsize=7, colors=DARK)
ax.spines[["top","right"]].set_visible(False)
ax.spines[["left","bottom"]].set_color("#C0CADC")
ax.tick_params(colors="#C0CADC", which="both")
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_color(DARK)

ax.set_title("Calibration Bound", fontsize=11, fontweight="bold",
             color=DARK, pad=5)

plt.tight_layout(pad=0.3)
plt.savefig(OUT, dpi=DPI, bbox_inches="tight", facecolor=CARD)
plt.close()
print(f"Saved → {OUT}")
