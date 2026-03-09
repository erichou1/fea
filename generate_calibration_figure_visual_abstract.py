"""
Generate a tiny-size-friendly calibration figure from SASTO held-out compliance data.

Single-panel figure with only:
    perfect-agreement line
    raw binned trend
    calibrated binned trend
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.isotonic import IsotonicRegression

OUT = "poster_images_extracted/fig_calibration_visual_abstract.png"
TP_PATH = Path("fea_ml/runs/v3/test_predictions.npz")

W, H = 5.8, 4.8
DPI = 300

NAVY = "#062B7A"
BLUE = "#0A3D9A"
ORANGE = "#E87C3E"
TEAL = "#008C9E"
GOLD = "#CFA535"
DARK = "#0B1736"
CARD = "#F7F9FC"
WHITE = "#FFFFFF"
GRAY = "#C0CADC"


def style_axes(ax):
    ax.set_facecolor(CARD)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRAY)
    ax.tick_params(colors=DARK, labelsize=8, length=3)
    ax.grid(True, color=GRAY, alpha=0.12, linewidth=0.7)


def main():
    tp = np.load(TP_PATH, allow_pickle=True)
    true_all = tp["true"][:, 2].astype(float)       # compliance
    pred_all = tp["pred_mean"][:, 2].astype(float)  # compliance

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(true_all))
    n_cal = len(true_all) // 2
    cal_idx = perm[:n_cal]
    val_idx = perm[n_cal:]

    pred_cal = pred_all[cal_idx]
    true_cal = true_all[cal_idx]
    pred_val = pred_all[val_idx]
    true_val = true_all[val_idx]

    iso = IsotonicRegression(y_min=0.0, out_of_bounds="clip")
    iso.fit(pred_cal, true_cal)
    fig, ax = plt.subplots(figsize=(W, H), facecolor=WHITE)
    fig.patch.set_facecolor(WHITE)
    style_axes(ax)

    lo = min(true_all.min(), pred_all.min())
    hi = max(true_all.max(), pred_all.max())
    pad = 0.04 * (hi - lo)
    lo -= pad
    hi += pad

    bins = np.quantile(pred_val, np.linspace(0, 1, 6))
    bins = np.unique(bins)
    mids, means_raw, means_cal = [], [], []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (pred_val >= b0) & (pred_val <= b1 if b1 == bins[-1] else pred_val < b1)
        if np.sum(m) < 8:
            continue
        mids.append(float(np.mean(pred_val[m])))
        means_raw.append(float(np.mean(true_val[m])))
        means_cal.append(float(np.mean(iso.predict(pred_val[m]))))

    ax.plot([lo, hi], [lo, hi], linestyle="--", color=GOLD, linewidth=2.2, zorder=1)
    ax.plot(
        mids, means_raw, color=ORANGE, linewidth=3.2, marker="o",
        markersize=6.5, markerfacecolor=ORANGE, markeredgecolor=WHITE,
        markeredgewidth=0.8, zorder=3
    )
    ax.plot(
        mids, means_cal, color=TEAL, linewidth=3.2, marker="o",
        markersize=6.5, markerfacecolor=TEAL, markeredgecolor=WHITE,
        markeredgewidth=0.8, zorder=4
    )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    thresh = 1.15
    if lo < thresh < hi:
        ticks = [round(lo, 2), thresh, round(hi, 2)]
    else:
        ticks = [round(lo, 2), round((lo + hi) / 2, 2), round(hi, 2)]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("Predicted", fontsize=9, color=DARK)
    ax.set_ylabel("True", fontsize=9, color=DARK)
    ax.set_title("Compliance calibration", fontsize=11, color=DARK, pad=8, fontweight="bold")

    ax.text(
        mids[0] + 0.01 * (hi - lo), means_raw[0] + 0.035 * (hi - lo),
        "raw", color=ORANGE, fontsize=8, fontweight="bold"
    )
    ax.text(
        mids[-2], means_cal[-2] - 0.055 * (hi - lo),
        "calibrated", color=TEAL, fontsize=8, fontweight="bold"
    )

    plt.tight_layout(pad=0.6)
    plt.savefig(OUT, dpi=DPI, bbox_inches="tight", facecolor=WHITE)
    print(f"✓ Saved: {OUT}")
    plt.close()


if __name__ == "__main__":
    main()
