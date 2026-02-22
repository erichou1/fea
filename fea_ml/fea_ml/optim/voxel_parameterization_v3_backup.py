"""
Voxel-based geometry parameterization for structural optimization.

Implements **spatially-varying surface erosion** using per-part
Euclidean Distance Transforms (EDT).  Each part type's EDT is
computed independently so that wall-floor / wall-roof junctions
don't contaminate thickness estimates (prevents the "three-layer"
interior-wall artifact).

*   Every part type can be thinned.  ``min_remaining_thickness``
    guarantees that the deepest core of each cross-section survives
    → no through-holes when min_remain + preserve ≥ max-filter
    overestimation (≤ 1 voxel with filter size 3).
*   A 3×3 plan-view grid per part-type gives 36 CMA-ES parameters.
*   High-resolution (256³) erosion is available for smooth STL export
    via ``apply_hires()``.

Parameter layout
────────────────
    [ext_zone_0 … ext_zone_8,  int_zone_0 … int_zone_8,
     roof_zone_0 … roof_zone_8,  floor_zone_0 … floor_zone_8]
     → 4 × 9 = 36 dims
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter, zoom

from fea_ml.geometry.voxelize import (
    PART_EXTERIOR_WALL,
    PART_INTERIOR_WALL,
    PART_ROOF,
    PART_FLOOR,
    PART_OTHER,
)

# Part types that the optimizer can erode (ordered)
_EROSION_PARTS = [
    PART_EXTERIOR_WALL,
    PART_INTERIOR_WALL,
    PART_ROOF,
    PART_FLOOR,
]

# Default preserve-margin per part type.
# preserve_margin = extra DT depth reserved beyond ``min_remaining_thickness``.
#   Together with min_remain they must exceed the maximum_filter
#   overestimation (~1 voxel for size=3) to guarantee no through-holes.
_DEFAULT_PRESERVE_MARGIN: Dict[int, float] = {
    PART_EXTERIOR_WALL: 0.15,   # thinning allowed, no through-holes
    PART_INTERIOR_WALL: 0.0,    # thinning allowed; optimizer self-limits
    PART_ROOF: 0.15,            # thinning allowed, no through-holes
    PART_FLOOR: 0.15,           # thinning allowed, no through-holes
}


@dataclass
class SurfaceErosionConfig:
    """Configuration for DT-based surface erosion."""

    # Plan-view grid resolution (N×N)
    grid_res: int = 3  # 3×3 = 9 zones

    # Upscale factor for high-res STL export (128 → 256 at scale=2)
    hires_scale: int = 2

    # Kernel size for ``maximum_filter`` to estimate local wall thickness
    max_filter_size: int = 5

    # Per-part preserve margin (see module docstring)
    preserve_margin: Dict[int, float] = field(
        default_factory=lambda: dict(_DEFAULT_PRESERVE_MARGIN)
    )

    # Minimum remaining EDT thickness each surface keeps (voxels, at 128³).
    # Prevents features from becoming thinner than 2 × this value.
    min_remaining_thickness: float = 1.0

    # Remove connected components smaller than this many voxels.
    # Set 0 to disable.  Prevents floating debris.
    remove_small_components: int = 100


# Backward-compatible alias (used by voxel_optimizer.py)
VoxelMaskedErosionConfig = SurfaceErosionConfig


# ─────────────────── main parameterization ───────────────────────

class SurfaceErosionParam:
    """
    Distance-transform-based spatially-varying surface erosion.

    Construction precomputes:
        * EDT of the 128³ occupancy grid
        * ``local_max_dt`` via maximum_filter — estimates half-thickness
          of the wall at every voxel
        * (lazily) the same at 256³ for STL export

    ``apply()``      modifies the 128³ occupancy according to CMA-ES params.
    ``apply_hires()`` applies the same erosion logic at 256³.

    Total dims = 4 × N²  (4 part types  ×  N×N plan-view zones).
    """

    def __init__(
        self,
        config: SurfaceErosionConfig,
        occ_128: np.ndarray,
        part_128: np.ndarray,
    ) -> None:
        self.config = config
        S = config.grid_res
        self._n = S
        self._n2 = S * S
        self._n_parts = len(_EROSION_PARTS)

        # ── precompute at 128³ ────────────────────────────────────
        self._occ_128 = (occ_128 > 0).astype(np.uint8)
        self._part_128 = part_128.astype(np.uint8)
        self._shape_128 = self._occ_128.shape  # (D, H, W)

        # ── per-part EDT + safe_depth (prevents cross-part contamination)
        self._part_dt_128: Dict[int, np.ndarray] = {}
        self._safe_depth_128: Dict[int, np.ndarray] = {}
        for pid in _EROSION_PARTS:
            part_mask = (self._part_128 == pid) & (self._occ_128 > 0)
            pdt = distance_transform_edt(part_mask).astype(np.float32)
            self._part_dt_128[pid] = pdt
            local_max = maximum_filter(
                pdt, size=config.max_filter_size
            ).astype(np.float32)
            preserve = config.preserve_margin.get(pid, 0.15)
            self._safe_depth_128[pid] = np.maximum(0.0, local_max - preserve)

        # ── high-res data (lazy – built on first apply_hires call) ─
        self._hires_ready = False
        self._occ_hi: Optional[np.ndarray] = None
        self._part_hi: Optional[np.ndarray] = None
        self._part_dt_hi: Optional[Dict[int, np.ndarray]] = None
        self._safe_depth_hi: Optional[Dict[int, np.ndarray]] = None
        self._shape_hi: Optional[tuple] = None

    # ──────────────────── lazy high-res init ──────────────────────

    def _ensure_hires(self) -> None:
        if self._hires_ready:
            return
        sc = self.config.hires_scale
        print(f"  [SurfaceErosionParam] building {sc}x hires …", flush=True)
        self._occ_hi = (
            zoom(self._occ_128.astype(np.float32), sc, order=0) > 0.5
        ).astype(np.uint8)
        self._part_hi = zoom(
            self._part_128.astype(np.float32), sc, order=0
        ).astype(np.uint8)

        # Per-part EDT at high resolution
        self._part_dt_hi = {}
        self._safe_depth_hi = {}
        for pid in _EROSION_PARTS:
            part_mask = (self._part_hi == pid) & (self._occ_hi > 0)
            pdt = distance_transform_edt(part_mask).astype(np.float32)
            self._part_dt_hi[pid] = pdt
            local_max = maximum_filter(
                pdt,
                size=self.config.max_filter_size * sc,
            ).astype(np.float32)
            preserve = self.config.preserve_margin.get(pid, 0.15)
            self._safe_depth_hi[pid] = np.maximum(0.0, local_max - preserve)

        self._shape_hi = self._occ_hi.shape
        self._hires_ready = True
        print("  [SurfaceErosionParam] hires precomputation done.", flush=True)

    # ──────────────────── public interface ────────────────────────

    def parameter_dim(self) -> int:
        return self._n_parts * self._n2

    def default_params(self) -> np.ndarray:
        """All zeros → no erosion (keep everything)."""
        return np.zeros(self.parameter_dim(), dtype=np.float32)

    def random_params(
        self, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(0.0, 1.0, size=self.parameter_dim()).astype(np.float32)

    # ──────────────────── apply at 128³ ───────────────────────────

    def apply(
        self,
        occ: np.ndarray,
        part: np.ndarray,
        edit_mask: np.ndarray,
        protected_mask: np.ndarray,
        params: np.ndarray,
    ) -> np.ndarray:
        """
        Apply DT-based erosion at 128³.

        For every solid voxel *v* in zone *z* with part *p*:

            safe_depth = max(0,  local_max_dt[v] − preserve_margin[p])
            threshold  = params[p, z] × safe_depth[v]
            keep v  ⟺  dt[v] > threshold

        Returns:
            (D, H, W)  uint8 occupancy grid.
        """
        return self._apply_impl(
            occ_ref=self._occ_128,
            part_ref=self._part_128,
            part_dt_map=self._part_dt_128,
            safe_depth_map=self._safe_depth_128,
            shape=self._shape_128,
            params=params,
            edit_mask=edit_mask,
            protected_mask=protected_mask,
            original_occ=occ,
            scale=1,
        )

    def apply_hires(self, params: np.ndarray) -> np.ndarray:
        """
        Apply erosion at 256³ (for STL export).

        Same erosion logic / parameters as ``apply()`` but at higher
        resolution.  No edit / protected masks are applied (those are
        128³-specific).
        """
        self._ensure_hires()
        return self._apply_impl(
            occ_ref=self._occ_hi,
            part_ref=self._part_hi,
            part_dt_map=self._part_dt_hi,
            safe_depth_map=self._safe_depth_hi,
            shape=self._shape_hi,
            params=params,
            edit_mask=None,
            protected_mask=None,
            original_occ=None,
            scale=self.config.hires_scale,
        )

    # ──────────────────── smooth erosion field ────────────────────

    def _make_smooth_field(
        self,
        params: np.ndarray,
        part_idx: int,
        H: int,
        W: int,
    ) -> np.ndarray:
        """
        Build a smooth (H, W) erosion-fraction field via bilinear
        interpolation of the N×N grid parameters.  Eliminates hard
        zone boundaries that cause slat / louver artifacts on thin
        structures.
        """
        gs = self._n
        start = part_idx * self._n2
        grid = params[start : start + self._n2].reshape(gs, gs).astype(np.float32)
        # Bilinear interpolation from (gs, gs) → (H, W)
        field = zoom(grid, (H / gs, W / gs), order=1, mode="nearest")
        return np.clip(field, 0.0, 1.0)

    # ──────────────────── core erosion ────────────────────────────

    def _apply_impl(
        self,
        occ_ref: np.ndarray,
        part_ref: np.ndarray,
        part_dt_map: Dict[int, np.ndarray],
        safe_depth_map: Dict[int, np.ndarray],
        shape: tuple,
        params: np.ndarray,
        edit_mask: Optional[np.ndarray],
        protected_mask: Optional[np.ndarray],
        original_occ: Optional[np.ndarray],
        scale: int = 1,
    ) -> np.ndarray:
        params = np.clip(params, 0.0, 1.0)
        D, H, W = shape
        result = occ_ref.copy().astype(np.uint8)

        min_remain = self.config.min_remaining_thickness * scale

        for part_idx, pid in enumerate(_EROSION_PARTS):
            part_mask = (part_ref == pid) & (occ_ref > 0)
            if not part_mask.any():
                continue
            safe_depth = safe_depth_map[pid]
            part_dt = part_dt_map[pid]

            # Smooth erosion field (H, W) → broadcast to (D, H, W)
            erosion_2d = self._make_smooth_field(params, part_idx, H, W)
            erosion_3d = np.broadcast_to(erosion_2d[None, :, :], (D, H, W))

            # Effective safe depth: reserve min_remain thickness per surface
            effective_sd = np.maximum(0.0, safe_depth - min_remain)

            threshold = erosion_3d * effective_sd
            remove = part_mask & (part_dt <= threshold)

            # Respect edit / protected masks
            if edit_mask is not None:
                remove = remove & (edit_mask > 0)
            if protected_mask is not None:
                remove = remove & (protected_mask == 0)

            result[remove] = 0

        # ── connected-component filtering ─────────────────────────
        min_comp = self.config.remove_small_components * (scale ** 3)
        if min_comp > 0:
            from scipy.ndimage import label
            labeled, num = label(result > 0)
            if num > 1:
                sizes = np.bincount(labeled.ravel())
                sizes[0] = 0  # ignore background
                keep = sizes >= min_comp
                result[~keep[labeled]] = 0

        # Ensure protected voxels are untouched
        if protected_mask is not None and original_occ is not None:
            result = np.where(protected_mask > 0, original_occ, result)

        return result

    # ──────────────────── helpers ─────────────────────────────────

    @staticmethod
    def compute_volume(occ: np.ndarray) -> int:
        return int(np.sum(occ > 0))


# Backward-compatible alias
VoxelMaskedErosion = SurfaceErosionParam
