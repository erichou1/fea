"""
Uncertainty estimation utilities for FEA surrogate models.

Supports MC Dropout and Deep Ensemble for conservative constraint checking.
"""
from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import nn


def enable_mc_dropout(model: nn.Module) -> None:
    """
    Enable dropout layers during inference for MC Dropout.
    
    Args:
        model: PyTorch model with dropout layers
    """
    if hasattr(model, "enable_mc_dropout"):
        model.enable_mc_dropout()
    else:
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()


def disable_mc_dropout(model: nn.Module) -> None:
    """
    Disable dropout for standard inference.
    
    Args:
        model: PyTorch model
    """
    if hasattr(model, "disable_mc_dropout"):
        model.disable_mc_dropout()
    else:
        model.eval()


def mc_predict(
    model: nn.Module,
    voxel: torch.Tensor,
    features: torch.Tensor,
    n_samples: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Make predictions with MC Dropout uncertainty.
    
    Args:
        model: Surrogate model
        voxel: (B, C, D, H, W) voxel input
        features: (B, F) feature input
        n_samples: Number of MC samples
        
    Returns:
        (mean, std) tensors each of shape (B, T)
    """
    enable_mc_dropout(model)
    
    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(voxel, features)
            predictions.append(pred)
    
    # Stack predictions: (N, B, T)
    stacked = torch.stack(predictions, dim=0)
    
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)
    
    return mean, std


def mc_predict_pointnet(
    model: nn.Module,
    points: torch.Tensor,
    features: torch.Tensor,
    n_samples: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MC Dropout for PointNet-style models.
    
    Args:
        model: PointNet model
        points: (B, N, 3) point cloud
        features: (B, F) features
        n_samples: Number of MC samples
        
    Returns:
        (mean, std) tensors
    """
    enable_mc_dropout(model)
    
    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(points, features)
            predictions.append(pred)
    
    stacked = torch.stack(predictions, dim=0)
    return stacked.mean(dim=0), stacked.std(dim=0)


def predict_with_uncertainty(
    model: nn.Module,
    voxel: torch.Tensor,
    features: torch.Tensor,
    method: str = "mc_dropout",
    n_samples: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unified interface for uncertainty estimation.
    
    Args:
        model: Surrogate model (single model or ensemble)
        voxel: Voxel input
        features: Feature input
        method: "mc_dropout" or "ensemble"
        n_samples: Number of samples (for MC Dropout)
        
    Returns:
        (mean, std) predictions
    """
    # Check if model is an ensemble
    if hasattr(model, "predict_with_uncertainty"):
        result = model.predict_with_uncertainty(voxel, features)
        return (
            torch.from_numpy(result.mean),
            torch.from_numpy(result.std),
        )
    
    # Use MC Dropout
    return mc_predict(model, voxel, features, n_samples)


def conservative_constraint_check(
    pred_mean: torch.Tensor,
    pred_std: torch.Tensor,
    target_idx: int,
    threshold: float,
    is_upper_bound: bool = True,
    k: float = 2.0,
) -> torch.Tensor:
    """
    Check constraint conservatively using uncertainty.
    
    For upper bounds (e.g., max displacement < limit):
        mean + k*std <= threshold
        
    For lower bounds (e.g., min safety factor >= limit):
        mean - k*std >= threshold
    
    Args:
        pred_mean: (B, T) mean predictions
        pred_std: (B, T) standard deviations
        target_idx: Index of target to check
        threshold: Constraint threshold
        is_upper_bound: True if constraint is "value <= threshold"
        k: Number of std deviations for conservative bound
        
    Returns:
        (B,) boolean tensor, True if constraint satisfied
    """
    mean = pred_mean[:, target_idx]
    std = pred_std[:, target_idx]
    
    if is_upper_bound:
        # mean + k*std <= threshold
        conservative_value = mean + k * std
        return conservative_value <= threshold
    else:
        # mean - k*std >= threshold
        conservative_value = mean - k * std
        return conservative_value >= threshold
