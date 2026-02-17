from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cma
import numpy as np
import torch

from fea_ml.models.uncertainty import mc_predict, enable_mc_dropout
from fea_ml.optim.parameterization import MeshModifier


@dataclass
class OptimizationResult:
    best_params: np.ndarray
    best_points: np.ndarray
    best_prediction: np.ndarray
    best_uncertainty: np.ndarray
    volume_reduction: float
    objective_history: list[float]


@dataclass
class OptimizationConfig:
    iterations: int = 40
    population_size: int = 12
    safety_factor_threshold: float = 1.5
    max_displacement_threshold: float = 1.0
    volume_weight: float = 1.0
    penalty_weight: float = 10.0
    mc_samples: int = 16


class SurrogateOptimizer:
    def __init__(
        self,
        model: torch.nn.Module,
        modifier: MeshModifier,
        config: OptimizationConfig,
        device: torch.device,
        target_mean: np.ndarray,
        target_std: np.ndarray,
    ) -> None:
        self.model = model
        self.modifier = modifier
        self.config = config
        self.device = device
        self.target_mean = target_mean
        self.target_std = target_std

    def optimize(
        self,
        baseline_points: np.ndarray,
        feature_vector: np.ndarray,
        target_indices: Dict[str, int],
    ) -> OptimizationResult:
        dim = self.modifier.parameter_dim()
        es = cma.CMAEvolutionStrategy(
            dim * [0.5],
            0.3,
            {"popsize": self.config.population_size, "maxiter": self.config.iterations},
        )
        objective_history: list[float] = []
        enable_mc_dropout(self.model)

        while not es.stop():
            solutions = es.ask()
            fitness = []
            for params in solutions:
                score, _ = self._evaluate_candidate(
                    baseline_points,
                    feature_vector,
                    target_indices,
                    np.array(params, dtype=np.float32),
                )
                fitness.append(score)
            es.tell(solutions, fitness)
            objective_history.append(min(fitness))

        best_params = np.array(es.result.xbest, dtype=np.float32)
        _, best_payload = self._evaluate_candidate(
            baseline_points,
            feature_vector,
            target_indices,
            best_params,
        )

        volume_reduction = 1.0 - best_payload["volume_proxy"] / best_payload["baseline_volume"]
        return OptimizationResult(
            best_params=best_params,
            best_points=best_payload["points"],
            best_prediction=best_payload["prediction"],
            best_uncertainty=best_payload["uncertainty"],
            volume_reduction=volume_reduction,
            objective_history=objective_history,
        )

    def _evaluate_candidate(
        self,
        baseline_points: np.ndarray,
        feature_vector: np.ndarray,
        target_indices: Dict[str, int],
        params: np.ndarray,
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        points = self.modifier.apply(baseline_points, params)
        volume_proxy = self.modifier.volume_proxy(points)
        baseline_volume = self.modifier.volume_proxy(baseline_points)

        points_tensor = torch.from_numpy(points[None, ...]).float().to(self.device)
        feature_tensor = torch.from_numpy(feature_vector[None, ...]).float().to(self.device)
        with torch.no_grad():
            mean, std = mc_predict(self.model, points_tensor, feature_tensor, self.config.mc_samples)
        prediction = mean.cpu().numpy().squeeze(0)
        uncertainty = std.cpu().numpy().squeeze(0)
        prediction = prediction * self.target_std + self.target_mean
        uncertainty = uncertainty * self.target_std

        sf_idx = target_indices["min_safety_factor"]
        disp_idx = target_indices["max_displacement"]
        sf_penalty = max(0.0, self.config.safety_factor_threshold - prediction[sf_idx])
        disp_penalty = max(0.0, prediction[disp_idx] - self.config.max_displacement_threshold)
        penalty = sf_penalty + disp_penalty

        objective = self.config.volume_weight * volume_proxy + self.config.penalty_weight * penalty
        return float(objective), {
            "points": points,
            "prediction": prediction,
            "uncertainty": uncertainty,
            "volume_proxy": volume_proxy,
            "baseline_volume": baseline_volume,
        }
