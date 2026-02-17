import numpy as np

from fea_ml.optim.parameterization import MeshModifier, MeshModifierConfig


def test_mesh_modifier_reduces_volume_proxy() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    modifier = MeshModifier(MeshModifierConfig(scale_min=0.7, scale_max=1.0))
    params = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    modified = modifier.apply(points, params)
    assert modifier.volume_proxy(modified) <= modifier.volume_proxy(points)
