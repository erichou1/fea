from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class MetricSummary:
    mae: Dict[str, float]
    rmse: Dict[str, float]
    constraint_accuracy: float


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: Tuple[str, ...]) -> MetricSummary:
    mae_vals = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse_vals = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    mae = {name: float(val) for name, val in zip(target_names, mae_vals)}
    rmse = {name: float(val) for name, val in zip(target_names, rmse_vals)}
    return MetricSummary(mae=mae, rmse=rmse, constraint_accuracy=0.0)


def constraint_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_indices: Dict[str, int],
    safety_factor_threshold: float,
    max_displacement_threshold: float,
) -> float:
    sf_idx = target_indices["min_safety_factor"]
    disp_idx = target_indices["max_displacement"]
    safe_true = (y_true[:, sf_idx] >= safety_factor_threshold) & (y_true[:, disp_idx] <= max_displacement_threshold)
    safe_pred = (y_pred[:, sf_idx] >= safety_factor_threshold) & (y_pred[:, disp_idx] <= max_displacement_threshold)
    return float(np.mean(safe_true == safe_pred))
