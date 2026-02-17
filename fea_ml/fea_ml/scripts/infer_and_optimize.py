from __future__ import annotations

import argparse
import json
from pathlib import Path

import meshio
import numpy as np
import torch

from fea_ml.data.dataset import normalize_points, sample_points
from fea_ml.models import SurrogatePointNet
from fea_ml.models.uncertainty import enable_mc_dropout, mc_predict
from fea_ml.optim import MeshModifier, MeshModifierConfig, OptimizationConfig, SurrogateOptimizer
from fea_ml.utils import load_config, resolve_device
from fea_ml.utils.io import load_normalization_stats


def load_mesh(path: str, normalize: bool) -> np.ndarray:
    mesh = meshio.read(path)
    points = mesh.points.astype(np.float32)
    if points.shape[1] > 3:
        points = points[:, :3]
    if normalize:
        points = normalize_points(points)
    return points, mesh


def load_features(path: str, config: dict) -> np.ndarray:
    payload = json.loads(Path(path).read_text())
    material_props = np.array(
        [
            payload["youngs_modulus"],
            payload["poisson_ratio"],
            payload["density"],
            payload["yield_stress"],
        ],
        dtype=np.float32,
    )
    material_onehot = np.zeros(len(config["materials"]), dtype=np.float32)
    material_onehot[config["materials"].index(payload["material_type"]) ] = 1.0
    load_onehot = np.zeros(len(config["load_cases"]), dtype=np.float32)
    load_onehot[config["load_cases"].index(payload["load_case"]) ] = 1.0
    return np.concatenate([material_props, material_onehot, load_onehot], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--baseline_mesh", required=True)
    parser.add_argument("--baseline_features", required=True)
    parser.add_argument("--output", default="runs/fea_surrogate/opt")
    args = parser.parse_args()

    config = load_config(args.config)
    stats = load_normalization_stats(args.normalization)
    target_mean = stats["target_mean"]
    target_std = stats["target_std"]
    feature_mean = stats["feature_mean"]
    feature_std = stats["feature_std"]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_points_full, baseline_mesh = load_mesh(
        args.baseline_mesh, config["data"].get("normalize_points", True)
    )
    baseline_points = sample_points(
        baseline_points_full,
        config["data"]["num_points"],
        seed=config["training"].get("seed", 0),
    )
    feature_vector = load_features(args.baseline_features, config)
    feature_vector = (feature_vector - feature_mean) / feature_std

    model = SurrogatePointNet(
        input_dim=baseline_points.shape[1],
        feature_dim=feature_vector.shape[0],
        target_dim=len(config["targets"]),
        dropout=config["model"]["dropout"],
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    device = resolve_device(prefer_gpu=True)
    model.to(device)

    modifier = MeshModifier(MeshModifierConfig(**config["optimization"]["parameterization"]))
    opt_config = OptimizationConfig(**config["optimization"]["search"])

    optimizer = SurrogateOptimizer(
        model=model,
        modifier=modifier,
        config=opt_config,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )

    target_indices = {name: i for i, name in enumerate(config["targets"])}
    result = optimizer.optimize(baseline_points, feature_vector, target_indices)

    best_full_points = modifier.apply(baseline_points_full, result.best_params)
    candidate_mesh = meshio.Mesh(points=best_full_points, cells=baseline_mesh.cells)
    meshio.write(output_dir / "candidate_mesh.vtk", candidate_mesh)
    (output_dir / "best_params.json").write_text(json.dumps(result.best_params.tolist(), indent=2))

    summary = {
        "volume_reduction": result.volume_reduction,
        "prediction": result.best_prediction.tolist(),
        "uncertainty": result.best_uncertainty.tolist(),
        "targets": config["targets"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    points_tensor = torch.from_numpy(baseline_points[None, ...]).float().to(device)
    feature_tensor = torch.from_numpy(feature_vector[None, ...]).float().to(device)
    enable_mc_dropout(model)
    with torch.no_grad():
        mean, std = mc_predict(model, points_tensor, feature_tensor, mc_samples=opt_config.mc_samples)
    mean = mean.cpu().numpy().squeeze(0) * target_std + target_mean
    std = std.cpu().numpy().squeeze(0) * target_std

    baseline_summary = {
        "prediction": mean.tolist(),
        "uncertainty": std.tolist(),
        "targets": config["targets"],
    }
    (output_dir / "baseline_summary.json").write_text(json.dumps(baseline_summary, indent=2))

    print("Optimization complete. Outputs saved to", output_dir)


if __name__ == "__main__":
    main()
