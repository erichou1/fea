"""Figure 1: AFDPS-style rounded task-card plate of the benchmark.

Layout follows the reference plate the author supplied: softly colored rounded
task cards on a white page, vertical task labels on the left rail, row titles in
a gutter, column headers with a metric line beneath, and a compact lattice of
scientific panels with white hairline separation.

Geometry rules taken from references/rounded-task-card-qualitative-plates.md:
fixed canvas, no bbox_inches='tight', independent horizontal zones for the
rail / gutter / lattice, and every card centered on its own lattice.

Content:
  Card A (blue)  one design along its trajectory, cutaway row + interior row
  Card B (peach) three other families at their deepest verified state

Part identity comes from the archive's part.npz and was verified geometrically:
part 1 sits at the footprint perimeter (77% within 3 voxels of the edge), part 2
is interior, part 3 is the top z band, part 4 is the thin base band. Note that
the campaign computes its own protection from min/max occupied x layers and does
NOT read protected_mask.npz, so protection is not part of the naming evidence.

Every voxel state is regenerated and checked against the frozen
state_occupancy_sha256 before drawing. Mismatch exits.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

sys.path.insert(0, "/Users/eric/workspace/fea-sasto-v/src")
from sasto.activity_campaign import geometric_trajectory
from sasto.g3_trajectory_calibration import family_seed

PAPER = Path("/Users/eric/workspace/sasto-modernization-control/v2/g4/paper")
FIGS = PAPER / "figures"
ARCHIVE = Path("/Users/eric/workspace/sasto-modernization-control/archives/fea_ml.zip")
INBOUND = Path("/Users/eric/workspace/sasto-g3-gb200-inbound/trajectory-calibration-gb200")

TRAJ = "00001"
ACROSS = ["00005", "00010", "00023", "00035"]
DEPTHS = ["(5,10%]", "(15,20%]", ">25%"]

PARTS = {
    1: ("Exterior walls", "#8fa3b8"),
    2: ("Interior partitions", "#cbbb98"),
    3: ("Roof", "#8a6a55"),
    4: ("Floor slabs", "#b6bcc2"),
}
REMOVED = "#c0392b"

CARD_A = "#e4edf6"
CARD_B = "#fbeadf"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8,
    "figure.dpi": 200,
})

# One type scale and one spacing scale, used everywhere. Earlier drafts mixed
# 7.4/7.6/8.2/8.4/9.0 and three different gaps, which is what read as untidy.
FS_TASK = 8.5     # vertical rail label
FS_HEAD = 8.0     # column header
FS_META = 7.5     # metric line and row titles
FS_KEY = 7.0      # legend

SP_HEAD = 0.026   # header baseline below card top
SP_META = 0.023   # metric line below header
SP_TOP = 0.020    # panel top below metric line
SP_ROW = 0.008    # between the two panel rows
SP_COL = -0.055   # NEGATIVE: the isometric projection of a building fills only
                  # ~70% of its panel box, so overlapping the boxes grows the
                  # drawn house without letting neighbours touch (measured gap
                  # stays ~0.15 in). Nudging zoom cannot beat that 70% cap.
SP_PAD = 0.014    # card inner padding

FW, FH = 7.2, 5.15


def member(z, sid, leaf):
    with np.load(io.BytesIO(z.read(f"fea_ml/data/runs_real/{sid}/{leaf}")),
                 allow_pickle=False) as loaded:
        return loaded["data"]


def record(sid):
    return json.loads((INBOUND / f"trajectory-development-{sid}.json").read_text())


def replay(sid, z):
    c = record(sid)
    base = member(z, sid, "occ.npz").astype(bool)
    parts = member(z, sid, "part.npz")
    _, states = geometric_trajectory(
        sample_id=sid, volume=base, batch_cap=40,
        ranking_seed=family_seed(c["family_id"]))
    for s in c["selected_states"]:
        got = hashlib.sha256(states[s["state_index"]].tobytes()).hexdigest()
        if got != s["state_occupancy_sha256"]:
            raise SystemExit(f"digest mismatch {sid} state {s['state_index']}")
    return base, parts, states, c


def tri(mask, color, alpha=1.0):
    """Shaded triangles for one mask. step_size=1: thin sheets break at 2."""
    if mask.sum() < 8:
        return None
    verts, faces, _, _ = marching_cubes(
        np.pad(mask.astype(np.float32), 1), level=0.5, step_size=1)
    t = verts[faces]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, nn, out=np.zeros_like(n), where=nn > 0)
    light = np.array([0.40, 0.58, 0.71])
    light /= np.linalg.norm(light)
    lam = 0.58 + 0.42 * np.clip(np.abs(n @ light), 0, 1)
    rgb = np.clip(np.array(matplotlib.colors.to_rgb(color))[None, :] * lam[:, None], 0, 1)
    return t, np.concatenate([rgb, np.full((len(rgb), 1), alpha)], axis=1)


def draw(ax, pieces):
    """One merged collection: matplotlib only depth-sorts within a collection."""
    ts = [p[0] for p in pieces if p]
    cs = [p[1] for p in pieces if p]
    if not ts:
        return
    m = Poly3DCollection(np.concatenate(ts), linewidths=0)
    m.set_facecolor(np.concatenate(cs))
    m.set_edgecolor("none")
    ax.add_collection3d(m)


def cut(v):
    keep = np.ones_like(v)
    nx, ny, _ = v.shape
    keep[nx // 2:, :ny // 2, :] = False
    return v & keep


def stage(ax, ref, elev=20, azim=-56):
    xs, ys, zs = np.where(ref)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 + 2
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(zs.min() - 1, zs.max() + 1)
    ax.set_box_aspect((1, 1, 0.62), zoom=1.58)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    # 3D axes force a square data aspect and silently shrink the rect passed to
    # add_axes; 'auto' makes the panel actually occupy the box we allocated.
    ax.set_aspect("auto")


def card(fig, x0, y0, x1, y1, color):
    """Card as a background AXES, not a fig-level patch.

    Figure-level patches paint above all axes regardless of zorder, so drawing
    the cards with fig.patches.append hid every 3D panel behind a pastel
    rectangle. A zorder-0 axes sits correctly underneath.
    """
    bg = fig.add_axes((x0, y0, x1 - x0, y1 - y0), zorder=0)
    bg.set_axis_off()
    bg.add_patch(FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0, transform=bg.transAxes,
        boxstyle="round,pad=0,rounding_size=0.045",
        facecolor=color, edgecolor="none", clip_on=False))
    return bg


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(ARCHIVE)
    prov = {"fea_ml_zip_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
            "part_semantics": {str(k): v[0] for k, v in PARTS.items()},
            "verified_states": {}}

    fig = plt.figure(figsize=(FW, FH))
    fig.patch.set_facecolor("white")

    # ---- zones: one margin everywhere -------------------------------
    M = 0.014                      # outer page margin
    RAIL = M + 0.020               # vertical task label
    GUT_X = RAIL + 0.016           # row-title gutter starts
    GUT_W = 0.070
    LAT_X = GUT_X + GUT_W + 0.008
    LAT_R = 1.0 - M - 0.008
    LAT_W = LAT_R - LAT_X

    # panels sized from the lattice, cards derived from the panels
    PW = (LAT_W - 3 * SP_COL) / 4
    PH = PW * FW / FH / 1.42   # panel aspect matched to the projected building
    HDR = SP_HEAD + SP_META + SP_TOP

    ay1 = 1.0 - M
    rowA0 = ay1 - HDR - PH
    rowA1 = rowA0 - SP_ROW - PH
    ay0 = rowA1 - SP_PAD

    by1 = ay0 - 0.018
    rowB0 = by1 - HDR - PH
    LEG_H = 0.030                  # legend strip inside card B
    by0 = rowB0 - LEG_H - SP_PAD

    # ---- Card A -----------------------------------------------------
    card(fig, M, ay0, 1.0 - M, ay1, CARD_A)
    fig.text(RAIL, (ay0 + ay1) / 2, "Erosion trajectory", rotation=90,
             ha="center", va="center", fontsize=FS_TASK, fontweight="bold")

    base, parts, states, c = replay(TRAJ, z)
    sel = {s["bin_label"]: s for s in c["selected_states"]}
    cols = [("baseline", None)] + [(b, sel[b]) for b in DEPTHS]
    row_y = [rowA0, rowA1]

    for ci, (label, s) in enumerate(cols):
        px = LAT_X + ci * (PW + SP_COL)
        frac = 0.0 if s is None else s["fraction_removed"] * 100
        fig.text(px + PW / 2, ay1 - SP_HEAD, label, ha="center", va="center",
                 fontsize=FS_HEAD, fontweight="bold" if s is None else "normal")
        fig.text(px + PW / 2, ay1 - SP_HEAD - SP_META,
                 f"{frac:.1f}% removed", ha="center", va="center",
                 fontsize=FS_META, color=REMOVED)

        v = base if s is None else states[s["state_index"]]
        gone = base & ~v

        ax = fig.add_axes((px, row_y[0], PW, PH), projection="3d", zorder=3)
        pieces = [tri(cut((parts == pid) & v), col)
                  for pid, (_, col) in PARTS.items()]
        if s is not None:
            pieces.append(tri(cut(gone), REMOVED, 0.80))
        draw(ax, pieces)
        stage(ax, cut(base))

        # interior row: partitions that survive, plus the ones erosion took,
        # so removal is visible here too rather than only as absence
        ax = fig.add_axes((px, row_y[1], PW, PH), projection="3d", zorder=3)
        ipieces = [tri(cut((parts == 2) & v), PARTS[2][1])]
        if s is not None:
            ipieces.append(tri(cut((parts == 2) & gone), REMOVED, 0.80))
        draw(ax, ipieces)
        stage(ax, cut((parts == 2) & base), elev=34, azim=-56)

        if s is not None:
            prov["verified_states"][label] = {
                "sample": TRAJ, "state_index": s["state_index"],
                "state_occupancy_sha256": s["state_occupancy_sha256"],
                "fraction_removed": s["fraction_removed"]}

    for yy, title in ((row_y[0], "Cutaway"), (row_y[1], "Interior only")):
        fig.text(GUT_X + GUT_W, yy + PH / 2, title, ha="right", va="center",
                 fontsize=FS_META, color="#333333")

    # ---- Card B -----------------------------------------------------
    card(fig, M, by0, 1.0 - M, by1, CARD_B)
    fig.text(RAIL, (by0 + by1) / 2, "Other families", rotation=90,
             ha="center", va="center", fontsize=FS_TASK, fontweight="bold")

    lat_b_x = LAT_X

    for ci, sid in enumerate(ACROSS):
        b2, p2, st, cc = replay(sid, z)
        deep = max(cc["selected_states"], key=lambda s: s["fraction_removed"])
        v = st[deep["state_index"]]
        px = lat_b_x + ci * (PW + SP_COL)
        fig.text(px + PW / 2, by1 - SP_HEAD, f"Sample {sid}", ha="center",
                 va="center", fontsize=FS_HEAD)
        fig.text(px + PW / 2, by1 - SP_HEAD - SP_META,
                 f"{deep['fraction_removed'] * 100:.1f}% removed",
                 ha="center", va="center", fontsize=FS_META, color=REMOVED)
        ax = fig.add_axes((px, rowB0, PW, PH), projection="3d", zorder=3)
        pieces = [tri(cut((p2 == pid) & v), col)
                  for pid, (_, col) in PARTS.items()]
        pieces.append(tri(cut(b2 & ~v), REMOVED, 0.80))
        draw(ax, pieces)
        stage(ax, cut(b2))
        prov["verified_states"][f"deep-{sid}"] = {
            "sample": sid, "state_index": deep["state_index"],
            "state_occupancy_sha256": deep["state_occupancy_sha256"],
            "fraction_removed": deep["fraction_removed"]}

    handles = [Patch(facecolor=cl, edgecolor="none", label=nm)
               for nm, cl in PARTS.values()]
    handles.append(Patch(facecolor=REMOVED, alpha=0.75, edgecolor="none",
                         label="Removed by erosion"))
    fig.legend(handles=handles, loc="center", ncol=5,
               bbox_to_anchor=(0.5, by0 + (LEG_H + SP_PAD) / 2),
               frameon=False, fontsize=FS_KEY, handlelength=0.85,
               handleheight=0.85, columnspacing=1.3, borderaxespad=0)

    fig.savefig(FIGS / "benchmark.pdf", facecolor="white")
    plt.close(fig)
    (FIGS / "benchmark-provenance.json").write_text(
        json.dumps(prov, indent=2, sort_keys=True) + "\n")
    print("wrote benchmark.pdf")
    print(f"  verified {len(prov['verified_states'])} states")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
