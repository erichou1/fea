"""Figure 1: the real eroded geometry, replayed and hash-verified.

Every voxel state drawn here is REGENERATED and then checked against the
state_occupancy_sha256 recorded in the frozen trajectory record. If a digest
does not match, the script exits rather than drawing.

This corrects an earlier assumption. The occupancy IS recoverable: the erosion
is deterministic given (baseline occupancy, sample_id, ranking_seed =
family_seed(family_id)), and the baseline comes from the hash-pinned
fea_ml.zip. The earlier replay failed only because it omitted ranking_seed.

Row 1  one family's trajectory, baseline -> deep, with the verified digest
Row 2  the same depth band across four different families
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "/Users/eric/workspace/fea-sasto-v/src")
from sasto.activity_campaign import geometric_trajectory
from sasto.g3_trajectory_calibration import family_seed

PAPER = Path("/Users/eric/workspace/sasto-modernization-control/v2/g4/paper")
FIGS = PAPER / "figures"
ARCHIVE = Path("/Users/eric/workspace/sasto-modernization-control/archives/fea_ml.zip")
INBOUND = Path("/Users/eric/workspace/sasto-g3-gb200-inbound/trajectory-calibration-gb200")

TRAJ_SAMPLE = "00001"
ACROSS = ["00005", "00010", "00023", "00044"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.titlesize": 8.5, "figure.dpi": 200,
})
ACCENT = "#b02418"


def record(sample_id: str) -> dict:
    return json.loads((INBOUND / f"trajectory-development-{sample_id}.json").read_text())


def replay(sample_id: str, archive: zipfile.ZipFile) -> tuple[np.ndarray, dict]:
    """Regenerate the trajectory and return (baseline, {state_index: volume})."""
    c = record(sample_id)
    raw = archive.read(f"fea_ml/data/runs_real/{sample_id}/occ.npz")
    with np.load(io.BytesIO(raw), allow_pickle=False) as loaded:
        base = loaded["data"].astype(bool)
    _, states = geometric_trajectory(
        sample_id=sample_id, volume=base, batch_cap=40,
        ranking_seed=family_seed(c["family_id"]),
    )
    # verify every selected state against its frozen digest
    for s in c["selected_states"]:
        si = s["state_index"]
        got = hashlib.sha256(states[si].tobytes()).hexdigest()
        if got != s["state_occupancy_sha256"]:
            raise SystemExit(f"digest mismatch {sample_id} state {si}; refusing to draw")
    return base, states


def draw_voxels(ax, vol: np.ndarray, title: str, sub: str = "",
                cutaway: bool = True, removed: np.ndarray | None = None) -> None:
    """Downsample 64^3 -> 32^3 by max-pooling so matplotlib can draw it.

    A front quadrant is cut away so interior erosion is visible; without it the
    outer shell hides almost every deletion and the trajectory looks static.
    Voxels removed relative to the baseline are drawn in red.
    """
    def pool(a):
        return (a[::2, ::2, ::2] | a[1::2, ::2, ::2] | a[::2, 1::2, ::2]
                | a[::2, ::2, 1::2] | a[1::2, 1::2, ::2] | a[1::2, ::2, 1::2]
                | a[::2, 1::2, 1::2] | a[1::2, 1::2, 1::2])

    v = pool(vol)
    gone = pool(removed) & ~v if removed is not None else np.zeros_like(v)

    if cutaway:
        nx, ny, _ = v.shape
        keep = np.ones_like(v)
        keep[nx // 2:, :ny // 2, :] = False
        v = v & keep
        gone = gone & keep

    shown = v | gone
    colors = np.zeros(shown.shape + (4,), dtype=float)
    zi = np.arange(shown.shape[2])
    frac = zi / max(1, shown.shape[2] - 1)
    for k in zi:
        g = 0.60 + 0.30 * frac[k]
        colors[:, :, k, :] = (g * 0.78, g * 0.85, g * 0.94, 1.0)
    colors[gone] = (0.69, 0.14, 0.09, 0.55)

    ax.voxels(shown, facecolors=colors, edgecolor=(0.22, 0.25, 0.30, 0.22),
              linewidth=0.1, shade=False)
    ax.set_box_aspect((1, 1, 0.8), zoom=1.30)
    ax.view_init(elev=22, azim=-52)
    ax.set_title(title, fontsize=8.5, pad=-1)
    if sub:
        ax.text2D(0.5, -0.045, sub, transform=ax.transAxes, ha="center",
                  fontsize=7, color=ACCENT)
    ax.grid(False)
    ax.set_axis_off()


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(ARCHIVE)
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    prov = {"fea_ml_zip_sha256": digest, "verified_states": {}}

    fig = plt.figure(figsize=(6.9, 4.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], hspace=0.30, wspace=0.02)

    # row 1: one family down its trajectory
    base, states = replay(TRAJ_SAMPLE, archive)
    c = record(TRAJ_SAMPLE)
    sel = {s["bin_label"]: s for s in c["selected_states"]}
    n0 = int(base.sum())

    ax = fig.add_subplot(gs[0, 0], projection="3d")
    draw_voxels(ax, base, "baseline", "0.0% removed")
    prov["verified_states"]["baseline"] = {"sample": TRAJ_SAMPLE, "voxels": n0}

    for col, blabel in enumerate(["(5,10%]", "(15,20%]", ">25%"], start=1):
        s = sel[blabel]
        v = states[s["state_index"]]
        ax = fig.add_subplot(gs[0, col], projection="3d")
        draw_voxels(ax, v, blabel, f"{s['fraction_removed'] * 100:.1f}% removed",
                    removed=base)
        prov["verified_states"][blabel] = {
            "sample": TRAJ_SAMPLE,
            "state_index": s["state_index"],
            "state_occupancy_sha256": s["state_occupancy_sha256"],
            "fraction_removed": s["fraction_removed"],
        }

    # row 2: the deepest band across four other families
    for col, sid in enumerate(ACROSS):
        b2, st = replay(sid, archive)
        cc = record(sid)
        deep = [s for s in cc["selected_states"] if s["bin_label"] == ">25%"]
        if not deep:
            deep = [max(cc["selected_states"], key=lambda s: s["fraction_removed"])]
        s = deep[0]
        ax = fig.add_subplot(gs[1, col], projection="3d")
        draw_voxels(ax, st[s["state_index"]], f"sample {sid}",
                    f"{s['fraction_removed'] * 100:.1f}% removed", removed=b2)
        prov["verified_states"][f"deep-{sid}"] = {
            "sample": sid,
            "state_index": s["state_index"],
            "state_occupancy_sha256": s["state_occupancy_sha256"],
            "fraction_removed": s["fraction_removed"],
        }

    fig.text(0.5, 0.955, "(a) one design along its erosion trajectory",
             ha="center", fontsize=9)
    fig.text(0.5, 0.468, "(b) the deepest band across four other families",
             ha="center", fontsize=9)
    fig.text(0.5, 0.012, "front quadrant cut away; red marks material removed "
                         "relative to the baseline",
             ha="center", fontsize=7.5, color="#555555")

    fig.savefig(FIGS / "benchmark.pdf", bbox_inches="tight")
    plt.close(fig)
    (FIGS / "benchmark-provenance.json").write_text(
        json.dumps(prov, indent=2, sort_keys=True) + "\n")
    print("wrote benchmark.pdf")
    print(f"  fea_ml.zip sha256 {digest[:16]}")
    print(f"  verified {len(prov['verified_states'])} states against frozen digests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
