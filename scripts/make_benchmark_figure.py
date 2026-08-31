"""Figure 1: the benchmark. Schematic load case + real source envelopes.

Two panels, and they make different kinds of claim, so they are drawn
differently and captioned differently.

(a) SCHEMATIC. The design domain, supports, load, and the erosion sequence.
    Drawn, not measured. No data is implied.

(b) REAL GEOMETRY. Source wireframe envelopes for four development-role
    families that appear in the measured population. Each file is verified by
    SHA-256 against v2/legacy-audit/source_lineage.csv before it is drawn, so
    what is plotted is provably the input geometry for that sample_id.

What panel (b) is NOT: the voxelized 64^3 occupancy the solver saw. That is
recorded in the trajectory records only as state_occupancy_sha256, and the
historical thickness sampling used a salted hash with no preserved
PYTHONHASHSEED, so the exact meshes are not regenerable. The caption says so.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

PAPER = Path("/Users/eric/workspace/sasto-modernization-control/v2/g4/paper")
FIGS = PAPER / "figures"
SRC = Path("/Users/eric/workspace/sasto-data/3dwire-546d66a-source")
LINEAGE = Path("/Users/eric/workspace/sasto-modernization-control/v2"
               "/legacy-audit/source_lineage.csv")

# development-role families present in the measured population
SAMPLES = ["00001", "00010", "00023", "00044"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.labelsize": 8.5, "axes.titlesize": 9, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 8, "axes.linewidth": 0.7,
    "figure.dpi": 200,
})
INK, ACCENT, MUTED, FILLED = "#1a1a1a", "#b02418", "#5a7fa6", "#c8d4e0"


def load_lineage() -> dict[str, dict]:
    with LINEAGE.open() as f:
        return {r["sample_id"]: r for r in csv.DictReader(f)}


def verified_wireframe(sample_id: str, lineage: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load a wireframe only after its file hash matches the lineage record."""
    rec = lineage[sample_id]
    path = SRC / f"{sample_id}.npz"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != rec["source_file_sha256"]:
        raise SystemExit(f"hash mismatch for {sample_id}; refusing to plot")
    d = np.load(path)
    return d["vertices"], d["lines"]


def draw_schematic(ax) -> None:
    """Design domain, supports, load, erosion. Drawn, not measured."""
    ax.set_xlim(-0.15, 3.25)
    ax.set_ylim(-0.62, 1.35)
    ax.axis("off")

    def box(x0, frac_removed, label, sub):
        w, h = 0.78, 0.78
        ax.add_patch(plt.Rectangle((x0, 0.18), w, h, facecolor=FILLED,
                                   edgecolor=INK, lw=0.9, zorder=2))
        # erosion: punch voids scaled by fraction removed
        rng = np.random.default_rng(7)
        n = int(frac_removed * 26)
        for _ in range(n):
            cx = x0 + 0.10 + rng.uniform(0, w - 0.20)
            cy = 0.28 + rng.uniform(0, h - 0.20)
            r = 0.030 + rng.uniform(0, 0.030)
            ax.add_patch(plt.Circle((cx, cy), r, facecolor="white",
                                    edgecolor="none", zorder=3))
        # support hatching along the base
        ax.plot([x0, x0 + w], [0.18, 0.18], color=INK, lw=1.6, zorder=4)
        for k in range(7):
            xs = x0 + 0.055 + k * (w - 0.11) / 6
            ax.plot([xs, xs - 0.055], [0.18, 0.10], color=INK, lw=0.7, zorder=4)
        # load arrow on the top face
        ax.annotate("", xy=(x0 + w / 2, 0.96), xytext=(x0 + w / 2, 1.24),
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.5),
                    zorder=5)
        ax.text(x0 + w / 2, 1.28, label, ha="center", va="bottom",
                fontsize=8, color=ACCENT if label == "F" else INK)
        ax.text(x0 + w / 2, 0.02, sub, ha="center", va="top", fontsize=7.5,
                color=INK)

    box(0.00, 0.00, "F", "baseline\n0% removed")
    box(1.15, 0.45, "F", "mid trajectory\n$\\sim$15% removed")
    box(2.30, 1.00, "F", "deep\n$>$25% removed")

    for x0 in (0.86, 2.01):
        ax.annotate("", xy=(x0 + 0.24, 0.57), xytext=(x0 - 0.04, 0.57),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6))
    ax.text(1.60, -0.52, "material removed along the trajectory",
            ha="center", fontsize=7.5, color=MUTED)


def draw_wire(ax, verts: np.ndarray, lines: np.ndarray, title: str) -> None:
    segs = [(verts[a], verts[b]) for a, b in lines]
    ax.add_collection3d(Line3DCollection(segs, colors=INK, linewidths=0.45,
                                         alpha=0.8))
    r = 0.80
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-0.55, 0.55)
    ax.set_box_aspect((1, 1, 0.62), zoom=1.42)
    ax.view_init(elev=26, azim=38)
    ax.set_title(title, fontsize=8, pad=-4)
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1, 1, 1, 0))
        pane.line.set_color((1, 1, 1, 0))
        pane.set_ticks([])


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    lineage = load_lineage()

    fig = plt.figure(figsize=(6.6, 4.05))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.92], hspace=0.42,
                          wspace=0.02)

    ax_s = fig.add_subplot(gs[0, :])
    draw_schematic(ax_s)
    ax_s.set_title("(a) load case and erosion trajectory (schematic)",
                   fontsize=9, pad=2)

    provenance = {}
    axes_b = []
    for i, sid in enumerate(SAMPLES):
        v, l = verified_wireframe(sid, lineage)
        ax = fig.add_subplot(gs[1, i], projection="3d")
        draw_wire(ax, v, l, f"sample {sid}")
        axes_b.append(ax)
        provenance[sid] = {
            "source_file_sha256": lineage[sid]["source_file_sha256"],
            "vertices": int(v.shape[0]),
            "lines": int(l.shape[0]),
        }

    # place the (b) header just above the top of the wireframe row, measured
    fig.canvas.draw()
    top_b = max(a.get_position().y1 for a in axes_b)
    fig.text(0.5, top_b + 0.055, "(b) source envelopes, four development "
                                 "families (hash-verified input geometry)",
             ha="center", va="bottom", fontsize=9)

    fig.savefig(FIGS / "benchmark.pdf", bbox_inches="tight")
    plt.close(fig)

    (FIGS / "benchmark-provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print("wrote benchmark.pdf")
    for sid, p in provenance.items():
        print(f"  {sid}  {p['vertices']:>4d} verts  {p['lines']:>4d} lines  "
              f"{p['source_file_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
