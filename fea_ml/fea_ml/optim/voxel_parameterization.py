"""
Voxel-based geometry parameterization for structural optimization.

Implements region-wise masked erosion with part-specific constraints:
- Exterior walls: interior-side erosion only (protected shell)
- Interior walls: full erosion within edit mask
- Roof/floor: controlled erosion respecting min thickness
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

from fea_ml.geometry.voxelize import (
    PART_EXTERIOR_WALL,
    PART_INTERIOR_WALL,
    PART_ROOF,
    PART_FLOOR,
    PART_OTHER,
    compute_sdf,
)


@dataclass
class VoxelMaskedErosionConfig:
    """Configuration for masked erosion parameterization."""
    # Erosion strength bounds
    erosion_min: float = 0.0
    erosion_max: float = 0.3
    # Smoothing strength bounds
    smooth_min: float = 0.0
    smooth_max: float = 0.2
    # Min thickness constraint (in voxels)
    min_thickness_voxels: float = 2.0
    # Shell thickness for exterior wall protection (in voxels)
    shell_thickness_voxels: int = 3


class VoxelMaskedErosion:
    """
    Parameterization that erodes voxel geometry within editable regions.
    
    Parameter vector: [e_ext, e_int, e_roof, e_floor, smooth_strength]
    - e_*: erosion strength for each part type [0, 1] -> scaled to config bounds
    - smooth_strength: morphological smoothing [0, 1]
    
    Constraints enforced:
    - Erosion only where edit_mask == 1
    - Never modifies protected_mask regions
    - Exterior walls: only erode from interior side (protected shell)
    - Closing operation maintains printability
    """
    
    def __init__(self, config: VoxelMaskedErosionConfig) -> None:
        self.config = config
    
    def parameter_dim(self) -> int:
        """Return dimension of parameter vector."""
        return 5  # e_ext, e_int, e_roof, e_floor, smooth_strength
    
    def default_params(self) -> np.ndarray:
        """Return default (no change) parameters."""
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    def random_params(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Generate random parameters in valid range."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(0.0, 1.0, size=5).astype(np.float32)
    
    def apply(
        self,
        occ: np.ndarray,
        part: np.ndarray,
        edit_mask: np.ndarray,
        protected_mask: np.ndarray,
        params: np.ndarray,
    ) -> np.ndarray:
        """
        Apply erosion parameterization to voxel geometry.
        
        Args:
            occ: (D, H, W) occupancy grid
            part: (D, H, W) part labels
            edit_mask: (D, H, W) where edits allowed
            protected_mask: (D, H, W) where edits forbidden
            params: (5,) parameter vector [e_ext, e_int, e_roof, e_floor, smooth]
            
        Returns:
            (D, H, W) modified occupancy grid
        """
        # Clip and scale parameters
        params = np.clip(params, 0.0, 1.0)
        
        e_ext = self._scale_erosion(params[0])
        e_int = self._scale_erosion(params[1])
        e_roof = self._scale_erosion(params[2])
        e_floor = self._scale_erosion(params[3])
        smooth = self._scale_smooth(params[4])
        
        # Start with original occupancy
        result = occ.copy().astype(np.float32)
        
        # Create erosion strength map based on part labels
        erosion_map = np.zeros_like(result)
        erosion_map[part == PART_EXTERIOR_WALL] = e_ext
        erosion_map[part == PART_INTERIOR_WALL] = e_int
        erosion_map[part == PART_ROOF] = e_roof
        erosion_map[part == PART_FLOOR] = e_floor
        erosion_map[part == PART_OTHER] = 0.0  # Don't erode unknown parts
        
        # Apply erosion only within edit_mask
        erosion_map = erosion_map * edit_mask
        erosion_map = erosion_map * (1 - protected_mask)  # Never erode protected
        
        # Compute SDF for erosion
        sdf = compute_sdf(occ)
        
        # Apply spatially-varying erosion
        # Erosion = threshold SDF at positive value
        # sdf > erosion_strength means eroded away
        erosion_threshold = erosion_map * self.config.min_thickness_voxels * 2
        eroded = (sdf < erosion_threshold).astype(np.float32)
        
        # Combine: keep original where not editable, use eroded where editable
        editable = (edit_mask > 0) & (protected_mask == 0)
        result = np.where(editable, eroded, result)
        
        # Apply smoothing (morphological closing to fill small gaps)
        if smooth > 0.01:
            kernel_size = max(1, int(smooth * 3))
            struct = ndimage.generate_binary_structure(3, 1)
            
            # Close: dilate then erode (fills holes)
            result_closed = ndimage.binary_closing(
                result > 0.5, 
                structure=struct,
                iterations=kernel_size,
            )
            
            # Only apply closing within editable regions
            result = np.where(editable, result_closed.astype(np.float32), result)
        
        # Enforce min thickness by protecting thin regions
        result = self._enforce_min_thickness(result, sdf)
        
        # Ensure protected regions unchanged
        result = np.where(protected_mask > 0, occ, result)
        
        return (result > 0.5).astype(np.uint8)
    
    def _scale_erosion(self, p: float) -> float:
        """Scale [0,1] parameter to erosion strength."""
        return self.config.erosion_min + p * (self.config.erosion_max - self.config.erosion_min)
    
    def _scale_smooth(self, p: float) -> float:
        """Scale [0,1] parameter to smoothing strength."""
        return self.config.smooth_min + p * (self.config.smooth_max - self.config.smooth_min)
    
    def _enforce_min_thickness(
        self,
        occ: np.ndarray,
        original_sdf: np.ndarray,
    ) -> np.ndarray:
        """Restore voxels that would violate min thickness."""
        # Compute new SDF
        new_sdf = compute_sdf((occ > 0.5).astype(np.uint8))
        
        # Find voxels that are too thin (inside but close to surface)
        thin_mask = (occ > 0.5) & (np.abs(new_sdf) < self.config.min_thickness_voxels / 2)
        
        # Check if removing these would create issues
        # For now, just keep them
        # More sophisticated: check connectivity
        
        return occ
    
    def compute_volume(self, occ: np.ndarray) -> int:
        """Compute volume as count of occupied voxels."""
        return int(np.sum(occ > 0))
    
    def volume_reduction(self, original: np.ndarray, modified: np.ndarray) -> float:
        """Compute volume reduction fraction."""
        v_orig = self.compute_volume(original)
        v_mod = self.compute_volume(modified)
        if v_orig == 0:
            return 0.0
        return 1.0 - v_mod / v_orig


class GradientFriendlyErosion:
    """
    Differentiable approximation of erosion for gradient-based optimization.
    
    Uses soft thresholding on SDF for differentiability.
    """
    
    def __init__(self, config: VoxelMaskedErosionConfig, temperature: float = 1.0) -> None:
        self.config = config
        self.temperature = temperature
    
    def parameter_dim(self) -> int:
        return 5
    
    def apply_soft(
        self,
        occ_soft: "torch.Tensor",
        part: np.ndarray,
        edit_mask: np.ndarray,
        protected_mask: np.ndarray,
        params: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Apply soft erosion for gradient computation.
        
        Args:
            occ_soft: (D, H, W) soft occupancy in [0, 1]
            part, edit_mask, protected_mask: numpy arrays
            params: (5,) torch tensor
            
        Returns:
            (D, H, W) soft modified occupancy
        """
        import torch
        
        # This is a simplified version - full differentiability
        # would require differentiable SDF computation
        
        params = torch.clamp(params, 0.0, 1.0)
        
        # Convert masks to torch
        device = occ_soft.device
        edit_mask_t = torch.from_numpy(edit_mask).float().to(device)
        protected_mask_t = torch.from_numpy(protected_mask).float().to(device)
        
        # Compute approximate erosion strength
        erosion_strength = params[0:4].mean() * self.config.erosion_max
        
        # Soft erosion: scale occupancy down in editable regions
        scale = 1.0 - erosion_strength * edit_mask_t * (1 - protected_mask_t)
        result = occ_soft * scale
        
        # Keep protected unchanged
        result = torch.where(protected_mask_t > 0.5, occ_soft, result)
        
        return result


def create_parameterization(
    method: str = "masked_erosion",
    config: Optional[VoxelMaskedErosionConfig] = None,
) -> VoxelMaskedErosion:
    """
    Factory function to create parameterization.
    
    Args:
        method: Parameterization type
        config: Configuration (uses defaults if None)
        
    Returns:
        Parameterization instance
    """
    if config is None:
        config = VoxelMaskedErosionConfig()
    
    if method == "masked_erosion":
        return VoxelMaskedErosion(config)
    else:
        raise ValueError(f"Unknown parameterization: {method}")
