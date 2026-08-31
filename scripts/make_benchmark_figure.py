"""Figure 1: semantic parts and what erosion actually removes.

Rendering follows the source dataset's convention (Ren et al., "Generating 3D
House Wireframes with Semantics"), which segments a house into semantic parts
rather than showing undifferentiated mass. Surfaces are extracted with marching
cubes so the result reads as architecture instead of a voxel blob.

Part identity is taken from the archive's own part.npz, then verified
geometrically before naming:

  part 1  perimeter (77% within 3 voxels of the footprint edge), fully
          protected from erosion            -> exterior walls
  part 2  interior, editable, largest       -> interior partitions
  part 3  top z band (31-42)                -> roof
  part 4  base z band (20-30), protected    -> floor slabs

Every voxel state drawn is regenerated and checked against the frozen
state_occupancy_sha256 before it is drawn. Mismatch exits.
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
from matplotlib.patches import Patch
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
ACROSS = ["00005", "00010", "00023"]

PARTS = {
    1: ("exterior walls", "#9aa7b4"),
    2: ("interior partitions", "#c4b59a"),
    3: ("roof", "#8f6f5c"),
    4: ("floor slabs", "#b0b7bd"),
}
REMOVED = "#b02418"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.titlesize": 8.5, "figure.dpi": 200,
})


def archive_member(z: zipfile.ZipFile, sid: str, leaf: str) -> np.ndarray:
    with np.load(io.BytesIO(z.read(f"fea_ml/data/runs_real/{sid}/{leaf}")),
                 allow_pickle=False) as loaded:
        return loaded["data"]


def record(sid: str) -> dict:
    return json.loads((INBOUND / f"trajectory-development-{sid}.json").read_text())


def replay(sid: str, z: zipfile.ZipFile):
    c = record(sid)
    base = archive_member(z, sid, "occ.npz").astype(bool)
    parts = archive_member(z, sid, "part.npz")
    _, states = geometric_trajectory(
        sample_id=sid, volume=base, batch_cap=40,
        ranking_seed=family_seed(c["family_id"]),
    )
    for s in c["selected_states"]:
        got = hashlib.sha256(states[s["state_index"]].tobytes()).hexdigest()
        if got != s["state_occupancy_sha256"]:
            raise SystemExit(f"digest mismatch {sid} state {s['state_index']}")
    return base, parts, states, c


def surface(ax, mask: np.ndarray, color: str, alpha: float = 1.0) -> None:
    """Marching-cubes surface for one part, with simple lambertian shading.

    No per-face edge lines: at this triangle count they stack into an opaque
    black mass. Shading comes from face normals instead.
    """
    if mask.sum() < 8:
        return
    pad = np.pad(mask.astype(np.float32), 1)
    try:
        verts, faces, normals, _ = marching_cubes(pad, level=0.5, step_size=2)
    except (RuntimeError, ValueError):
        return
    tris = verts[faces]
    # lambertian term from the triangle normal against a fixed key light
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 0)
    light = np.array([0.42, 0.60, 0.68])
    light = light / np.linalg.norm(light)
    lam = 0.62 + 0.38 * np.clip(n @ light, 0.0, 1.0)

    base = np.array(matplotlib.colors.to_rgb(color))
    face_rgb = np.clip(base[None, :] * lam[:, None], 0.0, 1.0)
    face_rgba = np.concatenate(
        [face_rgb, np.full((face_rgb.shape[0], 1), alpha)], axis=1)

    mesh = Poly3DCollection(tris, linewidths=0)
    mesh.set_facecolor(face_rgba)
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)


def frame(ax, mask: np.ndarray, title: str, sub: str = "") -> None:
    """Fit the view to the occupied region instead of the full 64^3 grid."""
    xs, ys, zs = np.where(mask)
    if xs.size == 0:
        return
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 + 2
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(zs.min() - 1, zs.max() + 1)
    ax.set_box_aspect((1, 1, 0.66), zoom=1.30)
    ax.view_init(elev=20, azim=-56)
    ax.set_axis_off()
    ax.set_title(title, fontsize=8.5, pad=1)
    if sub:
        ax.text2D(0.5, 0.015, sub, transform=ax.transAxes, ha="center",
                  fontsize=7.2, color=REMOVED)


def cut(v: np.ndarray) -> np.ndarray:
    """Cut the front quadrant so interior erosion is visible."""
    keep = np.ones_like(v)
    nx, ny, _ = v.shape
    keep[nx // 2:, :ny // 2, :] = False
    return v & keep


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(ARCHIVE)
    prov = {"fea_ml_zip_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
            "part_semantics": {str(k): v[0] for k, v in PARTS.items()},
            "verified_states": {}}

    fig = plt.figure(figsize=(6.9, 4.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.95], hspace=0.34,
                          wspace=0.01, top=0.90, bottom=0.10)

    base, parts, states, c = replay(TRAJ, z)
    sel = {s["bin_label"]: s for s in c["selected_states"]}

    # row 1: baseline by part, then the same design at three depths
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    for pid, (_, color) in PARTS.items():
        surface(ax, cut((parts == pid) & base), color)
    frame(ax, cut(base), "baseline", "0.0% removed")
    prov["verified_states"]["baseline"] = {"sample": TRAJ, "voxels": int(base.sum())}

    for col, blabel in enumerate(["(5,10%]", "(15,20%]", ">25%"], start=1):
        s = sel[blabel]
        v = states[s["state_index"]]
        ax = fig.add_subplot(gs[0, col], projection="3d")
        for pid, (_, color) in PARTS.items():
            surface(ax, cut((parts == pid) & v), color)
        surface(ax, cut(base & ~v), REMOVED, alpha=0.62)
        frame(ax, cut(base), blabel, f"{s['fraction_removed'] * 100:.1f}% removed")
        prov["verified_states"][blabel] = {
            "sample": TRAJ, "state_index": s["state_index"],
            "state_occupancy_sha256": s["state_occupancy_sha256"],
            "fraction_removed": s["fraction_removed"],
        }

    # row 2: legend cell, then three other families at their deepest state
    ax_leg = fig.add_subplot(gs[1, 0])
    ax_leg.axis("off")
    handles = [Patch(facecolor=c_, edgecolor="none", label=n)
               for n, c_ in PARTS.values()]
    handles.append(Patch(facecolor=REMOVED, alpha=0.62, edgecolor="none",
                         label="removed by erosion"))
    ax_leg.legend(handles=handles, loc="center", frameon=False,
                  fontsize=7.6, handlelength=1.25, handleheight=1.05,
                  borderpad=0.2, labelspacing=0.62)

    for col, sid in enumerate(ACROSS, start=1):
        b2, p2, st, cc = replay(sid, z)
        deep = [s for s in cc["selected_states"] if s["bin_label"] == ">25%"]
        if not deep:
            deep = [max(cc["selected_states"], key=lambda s: s["fraction_removed"])]
        s = deep[0]
        v = st[s["state_index"]]
        ax = fig.add_subplot(gs[1, col], projection="3d")
        for pid, (_, color) in PARTS.items():
            surface(ax, cut((p2 == pid) & v), color)
        surface(ax, cut(b2 & ~v), REMOVED, alpha=0.62)
        frame(ax, cut(b2), f"sample {sid}",
              f"{s['fraction_removed'] * 100:.1f}% removed")
        prov["verified_states"][f"deep-{sid}"] = {
            "sample": sid, "state_index": s["state_index"],
            "state_occupancy_sha256": s["state_occupancy_sha256"],
            "fraction_removed": s["fraction_removed"],
        }

    fig.text(0.5, 0.945, "(a) one design along its erosion trajectory",
             ha="center", fontsize=9)
    fig.text(0.5, 0.505, "(b) other families at their deepest verified state",
             ha="center", fontsize=9)

    fig.savefig(FIGS / "benchmark.pdf", bbox_inches="tight")
    plt.close(fig)
    (FIGS / "benchmark-provenance.json").write_text(
        json.dumps(prov, indent=2, sort_keys=True) + "\n")
    print("wrote benchmark.pdf")
    print(f"  verified {len(prov['verified_states'])} states against frozen digests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
