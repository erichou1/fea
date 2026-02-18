# Complete AI Optimization Training Pipeline

## Current Status Summary

✅ **Completed:**
- Batch meshing script with fallback algorithms
- Label preservation infrastructure  
- Batch FEA runner script
- Fixed mesh export to only include 3D elements (solves SfePy errors)

🔄 **In Progress:**
- None (ready to execute)

---

## Full Pipeline Execution Guide

### Prerequisites

1. **Activate fea environment** (required for FEA steps):
   ```bash
   conda activate fea
   ```

2. **Verify installations**:
   - Gmsh (for meshing)
   - SfePy (for FEA)
   - Pandas (for data aggregation)

---

### Phase 1: STEP → Mesh (3-4 hours for 20k files)

```bash
cd c:\Users\ericx\workspace\topopt_project\optimization\fea_gmsh_run

# Generate meshes from STEP files
python batch_step_to_mesh.py \
    --step-dir fea_out \
    --output-dir fea_out \
    --limit 20000 \
    --workers 8 \
    --h 0.10
```

**Output:** `*.msh` files with 3D tetrahedral meshes

**Expected success rate:** ~90% (18,000 meshes from 20,000 files)

---

### Phase 2: Mesh → FEA Results (50-100 hours for 20k files)

**IMPORTANT:** Ensure you're in the `fea` conda environment!

```bash
conda activate fea
cd c:\Users\ericx\workspace\topopt_project\optimization\fea_gmsh_run

# Run FEA simulations
python batch_run_fea.py \
    --mesh-dir fea_out \
    --output-dir fea_results \
    --workers 4 \
    --timeout 600
```

**What it does:**
- Runs structural analysis on each mesh
- Applies gravity, wind, seismic loads
- Calculates stress, displacement, safety factors
- Saves results to `fea_results/{sample_id}/fea_labels_combos.csv`
- Creates master file: `fea_results/master_fea_results.csv`

**Output structure:**
```
fea_results/
├── 00000/
│   ├── fea_labels_combos.csv  (FEA targets for this sample)
│   └── error.log              (if failed)
├── 00001/
│   ├── fea_labels_combos.csv
│   ...
└── master_fea_results.csv      (All results combined)
```

**Performance tuning:**
- Adjust `--workers` based on CPU cores (4-8 recommended)
- Increase `--timeout` if meshes are complex (default 600s = 10 min)
- Process can be stopped and resumed (skips completed)

---

### Phase 3: Prepare ML Training Data (1-2 hours)

```bash
cd c:\Users\ericx\workspace\topopt_project\fea_ml

# Combine geometry + FEA results into voxel format
python -m fea_ml.scripts.prepare_real_data \
    --parts-dir ../optimization/data/3dwire_parts_combined \
    --fea-dir ../optimization/fea_gmsh_run/fea_results \
    --output-dir data/runs \
    --resolution 64 \
    --workers 8
```

**What it does:**
- Voxelizes each STL part (exterior_walls, interior_rooms, roof, floor)
- Assigns part labels (1=exterior, 2=interior, 3=roof, 4=floor)
- Extracts FEA targets from `fea_labels_combos.csv`
- Generates edit/protected masks for optimization
- Saves to: `data/runs/{sample_id}/`

**Output per sample:**
```
data/runs/00000/
├── occ.npz              # 64³ occupancy grid (0/1)
├── part.npz             # 64³ part labels (0-5)
├── edit_mask.npz        # Editable voxels
├── protected_mask.npz   # Protected voxels
├── meta.json            # Materials, bounds, voxel size
└── targets.json         # FEA results (stress, displacement, etc.)
```

---

### Phase 4: Build Dataset Index (< 1 minute)

```bash
# Create train/val/test splits
python -m fea_ml.scripts.build_index \
    --runs-dir data/runs \
    --output data/manifests
```

**Output:**
- `data/manifests/train.json` (80% of samples)
- `data/manifests/val.json` (10% of samples)
- `data/manifests/test.json` (10% of samples)

---

### Phase 5: Train AI Surrogate Model (10-20 hours on GPU)

```bash
# Train 3D CNN ensemble
python -m fea_ml.scripts.train \
    --config configs/voxel_config.yaml \
    --output runs/experiment1
```

**What it trains:**
- 5-model deep ensemble (for uncertainty quantification)
- Predicts: stress, displacement, safety factor, compliance
- Input: 64³ voxel grid + part labels + material properties
- Output: Instant FEA predictions (milliseconds vs. hours)

**Monitoring:**
```bash
tensorboard --logdir runs/experiment1
```

**Expected training time:**
- Consumer GPU (RTX 3080/4090): ~15-20 hours for 18k samples
- H100: ~3-5 hours

---

### Phase 6: Evaluate Model (Optional but Recommended)

```bash
# Test on held-out data
python -m fea_ml.scripts.evaluate \
    --config configs/voxel_config.yaml \
    --checkpoint runs/experiment1/best.pt \
    --output runs/experiment1/eval
```

**Look for:**
- R² > 0.90 for all targets (good surrogate)
- Calibrated uncertainties (prediction intervals contain true values)

---

### Phase 7: Run Optimization 🎯

```bash
# Optimize a specific design
python -m fea_ml.scripts.optimize \
    --config configs/voxel_config.yaml \
    --checkpoint runs/experiment1/best.pt \
    --baseline data/runs/00000 \
    --output runs/experiment1/opt_00000
```

**What it does:**
- Takes baseline design from `data/runs/00000`
- Uses CMA-ES to search for optimal geometry
- Objective: Minimize material volume
- Constraints:
  - Safety factor ≥ 1.5
  - Displacement ≤ threshold
  - Compliance ≤ 1.05× baseline
- Runs 50 iterations with 16 candidates each

**Expected results:**
- 10-30% volume reduction
- All structural constraints satisfied
- ~1-2 hours per design on consumer GPU

---

## Time Estimates

| Phase | Description | Time (20k samples) |
|-------|-------------|-------------------|
| 1 | STEP → Mesh | 3-4 hours |
| 2 | Mesh → FEA | 50-100 hours |
| 3 | ML Data Prep | 1-2 hours |
| 4 | Build Index | < 1 minute |
| 5 | Train Model | 10-20 hours (GPU) |
| 6 | Evaluate | 30 minutes |
| 7 | Optimize (per design) | 1-2 hours |

**Total pipeline:** ~3-5 days continuous

---

## Troubleshooting

### Meshing Fails
- **Error:** HXT 3D mesh failed
- **Solution:** Script automatically tries Delaunay and Frontal algorithms

### FEA Fails
- **Error:** `region "XPlus" has no entities`
- **Solution:** Fixed! Updated gmsh_fragment_mesh.py to only export 3D elements

### FEA Timeout
- **Error:** Process timed out after 600 seconds
- **Solution:** Increase `--timeout` in batch_run_fea.py (e.g., `--timeout 1200`)

### Missing SfePy
- **Error:** `ModuleNotFoundError: No module named 'sfepy'`
- **Solution:** Activate fea environment: `conda activate fea`

### OOM (Out of Memory) during Training
- **Error:** CUDA out of memory
- **Solution:** Reduce batch_size in `configs/voxel_config.yaml` (default: 8 → try 4)

---

## Quick Start Commands (Copy-Paste Ready)

```bash
# 1. Meshing
cd c:\Users\ericx\workspace\topopt_project\optimization\fea_gmsh_run
python batch_step_to_mesh.py --step-dir fea_out --output-dir fea_out --limit 20000 --workers 8 --h 0.10

# 2. FEA (requires fea environment)
conda activate fea
python batch_run_fea.py --mesh-dir fea_out --output-dir fea_results --workers 4 --timeout 600

# 3. ML Data Prep
cd ../../fea_ml
python -m fea_ml.scripts.prepare_real_data --parts-dir ../optimization/data/3dwire_parts_combined --fea-dir ../optimization/fea_gmsh_run/fea_results --output-dir data/runs --resolution 64 --workers 8

# 4-7: Training & Optimization
python -m fea_ml.scripts.build_index --runs-dir data/runs --output data/manifests
python -m fea_ml.scripts.train --config configs/voxel_config.yaml --output runs/experiment1
python -m fea_ml.scripts.evaluate --config configs/voxel_config.yaml --checkpoint runs/experiment1/best.pt --output runs/experiment1/eval
python -m fea_ml.scripts.optimize --config configs/voxel_config.yaml --checkpoint runs/experiment1/best.pt --baseline data/runs/00000 --output runs/experiment1/opt_00000
```

---

## Next Steps

You are ready to start Phase 1 (Meshing). The 3D element export fix should resolve the FEA errors you were seeing.

**Recommended approach:**
1. Start with smaller batch (1000 samples) to validate entire pipeline
2. Once validated, scale to full 20k samples  
3. Run FEA in background (longest step)
4. Monitor progress in `fea_results/` directory
