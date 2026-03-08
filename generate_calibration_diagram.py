"""
Generate a simplified calibration-flow diagram for the Visual Abstract.

4-step horizontal flow (clean boxes, no scatter, no diamonds):
  [Voxel Field]  →  [Ensemble Predict]  →  [Conservative Bound]  →  [Commit or Undo]
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
EQbg  = "#E8EEF2"

OUT = "poster_images_extracted/icon_calibration.png"

# ── Canvas ───────────────────────────────────────────────────────────────────
W, H = 11.0, 3.2
fig, ax = plt.subplots(figsize=(W, H), facecolor=CARD)
ax.set_facecolor(CARD)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

# ── Layout constants ─────────────────────────────────────────────────────────
STEPS = [
    {
        "num": "1",
        "title": "Voxel Field",
        "eq":    None,
        "sub":   "5-member ensemble\noutput predictions",
        "fc":    LBLUE,
        "ec":    BLUE,
        "tc":    BLUE,
    },
    {
        "num": "2",
        "title": "Ensemble Predict",
        "eq":    r"$\mu_C,\ \sigma_C$",
        "sub":   "mean & std from\n5 forward passes",
        "fc":    EQbg,
        "ec":    BLUE,
        "tc":    BLUE,
    },
    {
        "num": "3",
        "title": "Conservative Bound",
        "eq":    r"$\hat{C}^+ = \mu + k\sigma$",
        "sub":   r"$k\!=\!1.0$ operating pt.",
        "fc":    "#FFF7E0",
        "ec":    GOLD,
        "tc":    DARK,
    },
    {
        "num": "4",
        "title": "Commit or Undo",
        "eq":    r"$\hat{C}^+/C_0 \leq 1.15$",
        "sub":   "accept step → commit\nfail → halve budget",
        "fc":    "#E8F8F5",
        "ec":    TEAL,
        "tc":    TEAL,
    },
]

N      = len(STEPS)
MARGIN = 0.35          # left/right page margin
ARROW  = 0.38          # horizontal gap for arrow
BH     = 1.72          # box height
BY     = H / 2         # box centre-y
BADGE  = 0.28          # step-badge radius
TITLE_H = 0.38         # bottom title bar height

# available width for boxes
usable = W - 2 * MARGIN - (N - 1) * ARROW
BW     = usable / N    # box width

def box_cx(i):
    return MARGIN + i * (BW + ARROW) + BW / 2

def draw_box(i, step):
    cx = box_cx(i)
    cy = BY

    # main card
    card = FancyBboxPatch(
        (cx - BW / 2, cy - BH / 2), BW, BH,
        boxstyle="round,pad=0.12",
        facecolor=step["fc"], edgecolor=step["ec"],
        linewidth=1.6, zorder=3,
    )
    ax.add_patch(card)

    # step badge (circle at top-left corner of box)
    badge_x = cx - BW / 2 + 0.02
    badge_y = cy + BH / 2 - 0.02
    badge = plt.Circle((badge_x, badge_y), BADGE,
                        facecolor=step["ec"], edgecolor="none", zorder=5)
    ax.add_patch(badge)
    ax.text(badge_x, badge_y, step["num"],
            ha="center", va="center", fontsize=8, color=WHITE,
            fontweight="bold", zorder=6)

    # title
    ax.text(cx, cy + BH / 2 - 0.42, step["title"],
            ha="center", va="center", fontsize=9.5, color=step["tc"],
            fontweight="bold", zorder=5)

    # equation (if present)
    if step["eq"]:
        ax.text(cx, cy + 0.10, step["eq"],
                ha="center", va="center", fontsize=11, color=DARK,
                fontweight="bold", zorder=5)

    # subtitle lines
    sub_y = cy - BH / 2 + 0.42
    for j, line in enumerate(step["sub"].split("\n")):
        ax.text(cx, sub_y + j * 0.26, line,
                ha="center", va="center", fontsize=7.5, color=DARK,
                fontstyle="italic", zorder=5)

def draw_arrow(i):
    """Draw gold arrow between box i and box i+1."""
    x0 = box_cx(i)     + BW / 2 + 0.04
    x1 = box_cx(i + 1) - BW / 2 - 0.04
    ax.annotate(
        "", xy=(x1, BY), xytext=(x0, BY),
        arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=2.2,
                        mutation_scale=16),
        zorder=4,
    )

# ── Draw all boxes and arrows ─────────────────────────────────────────────────
for i, step in enumerate(STEPS):
    draw_box(i, step)
    if i < N - 1:
        draw_arrow(i)

# ── Bottom title bar ──────────────────────────────────────────────────────────
bar = FancyBboxPatch(
    (0.0, 0.0), W, TITLE_H,
    boxstyle="round,pad=0.03",
    facecolor=NAVY, edgecolor="none", zorder=2,
)
ax.add_patch(bar)
ax.text(W / 2, TITLE_H / 2,
        "Calibration Pipeline  ·  Ensemble Uncertainty  →  Conservative Bound  →  Accept / Undo",
        ha="center", va="center", fontsize=8, color=WHITE,
        fontweight="bold", zorder=5)

# ── Gold rule above title bar ─────────────────────────────────────────────────
ax.plot([0, W], [TITLE_H + 0.01, TITLE_H + 0.01],
        color=GOLD, lw=1.4, zorder=3)

plt.tight_layout(pad=0.0)
plt.savefig(OUT, dpi=280, bbox_inches="tight", facecolor=CARD)
plt.close()
print(f"Saved → {OUT}")
