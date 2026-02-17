# FEA ML Surrogate + Voxel-Based Structural Optimization

End-to-end AI system for FEA-guided structural optimization of 3D printed housing components.

## Approach

**Two-stage surrogate + optimizer (Option C):**
1. **Surrogate Model**: 3D CNN predicts FEA metrics from voxel geometry
2. **CMA-ES Optimizer**: Searches for lower-volume designs satisfying structural constraints

## Quick Start

### Installation

```bash
cd fea_ml
pip install -r requirements.txt
```

### Generate Synthetic Data (for testing)

```bash
python -m fea_ml.scripts.generate_synthetic_data --output data/runs_test --n-samples 100
```

### Build Dataset Index

```bash
python -m fea_ml.scripts.build_index --runs-dir data/runs --output data/manifests
```

### Train Surrogate

```bash
python -m fea_ml.scripts.train --config configs/voxel_config.yaml --output runs/experiment1
```

### Evaluate Model

```bash
python -m fea_ml.scripts.evaluate \
    --config configs/voxel_config.yaml \
    --checkpoint runs/experiment1/best.pt \
    --output runs/experiment1/eval
```

### Run Optimization

```bash
python -m fea_ml.scripts.optimize \
    --config configs/voxel_config.yaml \
    --checkpoint runs/experiment1/best.pt \
    --baseline data/runs/sample_001 \
    --output runs/experiment1/optimization
```

## Data Format

Each simulation run should be stored in `data/runs/<run_id>/` with:

| File | Description |
|------|-------------|
| `occ.npz` | Occupancy grid `(D,H,W)` uint8 `{0,1}` |
| `sdf.npz` | Signed distance field `(D,H,W)` float32 (optional) |
| `part.npz` | Part labels `(D,H,W)` uint8 `{0-5}` |
| `edit_mask.npz` | Editable regions `(D,H,W)` uint8 |
| `protected_mask.npz` | Protected regions `(D,H,W)` uint8 |
| `mesh.msh` | Gmsh mesh with physical groups |
| `physical_groups.json` | Bidirectional part→physical_id mapping |
| `meta.json` | `{E, nu, density, yield_stress, material_label, load_case_id, length_unit, baseline_run_id}` |
| `targets.json` | `{max_von_mises, max_displacement, min_safety_factor, compliance}` |

### Part Labels
- 0: Empty
- 1: Exterior wall
- 2: Interior wall
- 3: Roof
- 4: Floor
- 5: Other/support

## Integration with Existing Pipeline

### After FEA Simulation

```python
from fea_ml.geometry.voxelize import stl_to_voxels, save_voxel_grids, generate_masks, VoxelizationConfig

# 1. Voxelize geometry
occ, bounds, voxel_size = stl_to_voxels("path/to/house.stl", resolution=64)

# 2. Assign part labels (from physical_groups.json)
part = assign_part_labels(occ, physical_groups)

# 3. Generate masks
config = VoxelizationConfig(resolution=64, shell_thickness_voxels=3)
edit_mask, protected_mask = generate_masks(occ, part, config)

# 4. Save to run directory
save_voxel_grids(VoxelGrids(occ, sdf, part, edit_mask, protected_mask, bounds, voxel_size), "data/runs/run_001")

# 5. Save metadata and targets
save_meta({"E": 2e11, "nu": 0.3, ...}, "data/runs/run_001/meta.json")
save_targets({"max_von_mises": fea_results.max_vm, ...}, "data/runs/run_001/targets.json")
```

### Retraining Recipe

```bash
# 1. Append new runs to data/runs/
# 2. Rebuild index
python -m fea_ml.scripts.build_index --runs-dir data/runs --output data/manifests

# 3. Retrain (or finetune with --resume)
python -m fea_ml.scripts.train --config configs/voxel_config.yaml --output runs/v2
```

## Model Architecture

- **3D CNN Encoder**: 4-layer convolutional encoder with batch norm
- **Global Pool**: Adaptive average pooling to fixed-size embedding
- **Feature MLP**: Processes material/load case features
- **Prediction Head**: Combined embedding → FEA targets

### Uncertainty Estimation

- **Deep Ensemble**: N=5 independent models (default)
- **MC Dropout**: Alternative for single-model inference
- **Conservative Constraints**: `mean - k*std ≥ threshold` for safety

## Optimization

### Parameterization

5-dimensional parameter vector:
- `e_ext`: Erosion strength for exterior walls
- `e_int`: Erosion strength for interior walls
- `e_roof`: Erosion strength for roof
- `e_floor`: Erosion strength for floor
- `smooth`: Morphological smoothing strength

### Constraints

| Constraint | Type | Default |
|------------|------|---------|
| Safety Factor | ≥ threshold | 1.5 |
| Displacement | ≤ threshold | 1.0 |
| Compliance | ≤ 1.05× baseline | - |

### Validity Checks

- **Watertight**: Exterior walls must not have through-holes
- **Min Thickness**: All regions ≥ 2 voxels thick
- **Connectivity**: Single connected component

## Tests

```bash
pytest fea_ml/fea_ml/tests -v
```

## GPU Support

- **Consumer GPU (12-24GB)**: Default 64³ resolution, batch_size=8
- **H100**: 128³ resolution, batch_size=32, ensemble parallel training

Enable mixed precision (AMP) in config for ~30% speedup:
```yaml
training:
  mixed_precision: true
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| OOM | Resolution too high | Reduce `resolution` or `batch_size` |
| Missing labels | Physical groups not exported | Check `physical_groups.json` exists |
| Invalid designs | Erosion too aggressive | Reduce `erosion_max` in config |
| Poor calibration | Model overconfident | Increase ensemble size or dropout |
