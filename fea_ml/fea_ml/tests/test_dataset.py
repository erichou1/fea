import json
from pathlib import Path

import meshio
import numpy as np

from fea_ml.data.dataset import FEADataset


def test_dataset_loading(tmp_path: Path) -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    cells = [("tetra", np.array([[0, 1, 2, 3]], dtype=np.int32))]
    mesh = meshio.Mesh(points=points, cells=cells)
    mesh_path = tmp_path / "sample.vtk"
    meshio.write(mesh_path, mesh)

    manifest_path = tmp_path / "manifest.jsonl"
    record = {
        "mesh_path": str(mesh_path),
        "youngs_modulus": 2.1e9,
        "poisson_ratio": 0.35,
        "density": 1200.0,
        "yield_stress": 35e6,
        "material_type": "mortar",
        "load_case": "case_a",
        "max_von_mises": 10.0,
        "max_displacement": 0.2,
        "min_safety_factor": 2.0,
        "compliance": 0.05,
    }
    manifest_path.write_text(json.dumps(record) + "
")

    dataset = FEADataset(
        str(manifest_path),
        target_names=("max_von_mises", "max_displacement", "min_safety_factor", "compliance"),
        material_types=("concrete", "mortar"),
        load_cases=("case_a", "case_b"),
        num_points=8,
        normalize=True,
        seed=123,
    )
    sample = dataset[0]
    assert sample["points"].shape == (8, 3)
    assert sample["features"].shape[0] == 4 + 2 + 2
    assert sample["targets"].shape[0] == 4
