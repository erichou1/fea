"""
V7: Wall-segment removal optimization.

At 128^3, building walls are only 1-2 voxels thick — too thin for
voxel-level thinning without creating holes/slats/double-wall artifacts.

This module segments interior walls into discrete wall segments via
2D skeleton analysis and makes binary keep/remove decisions per segment.

Key properties:
  - Exterior walls: FULLY PROTECTED (never modified)
  - Roof:           FULLY PROTECTED
  - Floor:          FULLY PROTECTED
  - Interior walls: segmented into ~29 individual wall segments
  - Junction voxels (where walls meet): ALWAYS KEPT
  - Each segment: binary keep/remove (1 CMA-ES parameter each)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from scipy.signal import convolve2d

PART_EXTERIOR_WALL = 1
PART_INTERIOR_WALL = 2
PART_ROOF = 3
PART_FLOOR = 4


@dataclass
class WallSegmentConfig:
    """Configuration for wall-segment removal optimization."""
    min_segment_pixels: int = 3   # minimum skeleton pixels per valid segment


class WallSegmentParam:
    """
    Binary keep/remove optimization of interior wall segments.

    Segments interior walls by:
      1. Project interior walls to XY (plan view)
      2. Skeletonize to find wall centerlines
      3. Detect junction points (>2 skeleton neighbours)
      4. Dilate junctions → protected junction zone
      5. Split skeleton at junctions → individual segments
      6. Voronoi-assign all wall pixels to nearest segment
      7. Extend 2D assignment to 3D column-wise

    CMA-ES controls one [0,1] parameter per segment.
    params[i] > 0.5  →  remove segment (i+1)
    params[i] ≤ 0.5  →  keep segment (i+1)
    """

    def __init__(
        self,
        config: WallSegmentConfig,
        occ: np.ndarray,
        part: np.ndarray,
    ) -> None:
        self.shape = occ.shape
        self.config = config
        self._build(occ, part)

    # ── segmentation ─────────────────────────────────────────────

    def _build(self, occ: np.ndarray, part: np.ndarray) -> None:
        int_mask = (part == PART_INTERIOR_WALL) & (occ > 0)

        # ---- 2D projection (plan view) ----
        proj_xy = int_mask.max(axis=2).astype(bool)
        print(f"  Interior wall 2D projection: {int(proj_xy.sum())} pixels",
              flush=True)

        # ---- Skeletonize ----
        skel = skeletonize(proj_xy)
        print(f"  Skeleton pixels: {int(skel.sum())}", flush=True)

        # ---- Junction detection ----
        kernel = np.ones((3, 3), dtype=int)
        kernel[1, 1] = 0
        nc = convolve2d(skel.astype(int), kernel, mode='same', boundary='fill')
        junctions = skel & (nc > 2)
        junction_zone = ndimage.binary_dilation(junctions, iterations=1)
        print(f"  Junction points: {int(junctions.sum())}, "
              f"zone pixels: {int(junction_zone.sum())}", flush=True)

        # ---- Split skeleton at junctions ----
        skel_split = skel.copy()
        skel_split[junction_zone] = False
        labeled_skel, n_raw = ndimage.label(skel_split)

        # ---- Filter small fragments ----
        min_px = self.config.min_segment_pixels
        valid = [i for i in range(1, n_raw + 1)
                 if (labeled_skel == i).sum() >= min_px]
        relabeled = np.zeros_like(labeled_skel)
        for new_id, old_id in enumerate(valid, 1):
            relabeled[labeled_skel == old_id] = new_id
        labeled_skel = relabeled
        n_seg = len(valid)
        print(f"  Valid segments after filtering (<{min_px}px): {n_seg}",
              flush=True)

        # ---- Voronoi assignment of all wall pixels to segments ----
        assignment_2d = np.zeros(proj_xy.shape, dtype=np.int32)
        assignment_2d[labeled_skel > 0] = labeled_skel[labeled_skel > 0]

        unlabeled = proj_xy & (labeled_skel == 0)
        if unlabeled.any():
            labeled_mask = labeled_skel > 0
            if labeled_mask.any():
                src = np.argwhere(labeled_mask)
                src_vals = labeled_skel[labeled_mask]
                tree = cKDTree(src)
                dst = np.argwhere(unlabeled)
                _, idx = tree.query(dst)
                assignment_2d[unlabeled] = src_vals[idx]

        # ---- Protect junction zone pixels (label → 0 = always kept) ----
        junction_wall = junction_zone & proj_xy
        assignment_2d[junction_wall] = 0

        # ---- Extend 2D assignment to 3D (column-wise) ----
        self.segment_3d = np.zeros(self.shape, dtype=np.int32)
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                seg = assignment_2d[x, y]
                if seg > 0:
                    zs = np.where(int_mask[x, y, :])[0]
                    if len(zs) > 0:
                        self.segment_3d[x, y, zs] = seg

        # ---- Statistics ----
        self.n_segments = n_seg
        total_int = int(int_mask.sum())
        total_seg = int((self.segment_3d > 0).sum())
        junction_count = total_int - total_seg

        self.segment_sizes: Dict[int, int] = {}
        for s in range(1, n_seg + 1):
            self.segment_sizes[s] = int((self.segment_3d == s).sum())

        print(f"  === Wall Segment Summary ===", flush=True)
        print(f"  Total interior wall voxels: {total_int}", flush=True)
        print(f"  Removable (in segments):    {total_seg} "
              f"({total_seg / max(total_int, 1) * 100:.1f}%)", flush=True)
        print(f"  Junction (protected):       {junction_count} "
              f"({junction_count / max(total_int, 1) * 100:.1f}%)", flush=True)
        print(f"  Segments:", flush=True)
        sorted_segs = sorted(self.segment_sizes.items(),
                             key=lambda x: x[1], reverse=True)
        for seg_id, cnt in sorted_segs:
            pct = cnt / max(total_int, 1) * 100
            print(f"    seg {seg_id:2d}: {cnt:5d} voxels ({pct:.1f}%)",
                  flush=True)

    # ── public interface ─────────────────────────────────────────

    def parameter_dim(self) -> int:
        """Number of CMA-ES parameters (one per wall segment)."""
        return self.n_segments

    def apply(
        self,
        occ: np.ndarray,
        part: np.ndarray,
        edit_mask: np.ndarray,
        protected_mask: np.ndarray,
        params: np.ndarray,
    ) -> np.ndarray:
        """
        Apply wall segment removal.

        params[i] > 0.5 → remove segment (i+1)
        params[i] ≤ 0.5 → keep segment (i+1)

        Exterior walls, roof, floor: UNTOUCHED.
        Junction voxels: ALWAYS KEPT.
        """
        result = occ.copy()
        for i in range(self.n_segments):
            if params[i] > 0.5:
                result[self.segment_3d == (i + 1)] = 0
        return result

    def removed_segment_ids(self, params: np.ndarray) -> List[int]:
        """Return list of segment IDs that would be removed."""
        return [i + 1 for i in range(self.n_segments) if params[i] > 0.5]

    def kept_segment_ids(self, params: np.ndarray) -> List[int]:
        """Return list of segment IDs that would be kept."""
        return [i + 1 for i in range(self.n_segments) if params[i] <= 0.5]

    def removal_summary(self, params: np.ndarray) -> Dict:
        """Return dict with removal details."""
        removed = self.removed_segment_ids(params)
        kept = self.kept_segment_ids(params)
        vox_removed = sum(self.segment_sizes.get(s, 0) for s in removed)
        vox_kept = sum(self.segment_sizes.get(s, 0) for s in kept)
        return {
            "removed_segments": removed,
            "kept_segments": kept,
            "n_removed": len(removed),
            "n_kept": len(kept),
            "n_total": self.n_segments,
            "voxels_removed": vox_removed,
            "voxels_kept": vox_kept,
        }

    @staticmethod
    def compute_volume(occ: np.ndarray) -> int:
        return int(np.sum(occ > 0))


# Backward-compatible aliases so old imports don't break
SurfaceErosionConfig = WallSegmentConfig
SurfaceErosionParam = WallSegmentParam
VoxelMaskedErosionConfig = WallSegmentConfig
VoxelMaskedErosion = WallSegmentParam
