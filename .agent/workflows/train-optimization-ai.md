---
description: Complete pipeline to train and run the optimization AI
---

# Complete AI Optimization Training Pipeline

Starting point: You have STEP files and wireframes ready.

## Phase 1: Engineering Simulation (STEP → FEA Results)

### 1. Convert STEP to Mesh (.msh)

**Location:** `c:\Users\ericx\workspace\topopt_project\optimization\fea_gmsh_run`

```bash
python gmsh_fragment_mesh.py --step-dir fea_out --output-dir fea_out --overwrite
```

**What it does:**
- Converts all `.step` files to `.msh` (Gmsh tetrahedral mesh)
- Exports `physical_groups.json` (part labels: exterior_walls, interior_rooms, roof, floor)
- This may take several hours for 53k+ files

**Output per sample:**
- `{id}_parts.msh` - mesh file
- `{id}_parts_physical_groups.json` - part-to-mesh mapping

---

### 2. Run FEA Simulations

```bash
python solve_asce7_22_asd_sfepy_ai_labels.py --mesh-dir fea_out --output-dir fea_out
```

**What it does:**
- Runs finite element analysis on each mesh
- Applies ASCE 7-22 load combinations (gravity, wind, combined)
- Computes stress, displacement, von Mises, compliance, etc.

**Output per sample:**
- `{id}_u.vtk` - displacement field visualization
- `fea_labels_combos.csv` - aggregated FEA targets
  - `max_u` (max displacement)
  - `max_von_mises` (peak stress)
  - `max_sigma_1`, `min_sigma_3` (principal stresses)
  - `strain_energy`, `compliance_like`

**Critical:** You need to organize the CSV outputs per-sample or modify the pipeline to track which results belong to which mesh.

---

## Phase 2: ML Data Preparation (FEA → Voxels)

### 3. Prepare Voxelized Training Data

**Location:** `c:\Users\ericx\workspace\topopt_project\fea_ml`

The current `prepare_real_data.py` expects separate STL part files, but your pipeline uses complete meshes with physical groups. You have two options:

#### Option A: Extract STL parts from meshes (Recommended)
You'll need to create a script that:
1. Reads each `.msh` file
2. Reads corresponding `physical_groups.json`
3. Extracts each part (exterior_walls, interior_rooms, roof, floor) as separate STL
4. Saves as `{id}_exterior_walls.stl`, `{id}_interior_rooms.stl`, etc.

#### Option B: Modify `prepare_real_data.py` to work with complete meshes
Update the script to voxelize from `.msh` directly using physical groups for labels.

**After you have part STLs:**

```bash
cd c:\Users\ericx\workspace\topopt_project\fea_ml

python -m fea_ml.scripts.prepare_real_data \
    --parts-dir C:\Users\ericx\workspace\topopt_project\optimization\fea_gmsh_run\fea_out \
    --fea-dir C:\Users\ericx\workspace\topopt_project\optimization\fea_gmsh_run\fea_out \
    --output-dir data/runs \
    --resolution 64 \
    --workers 8
```

**What it does:**
- Voxelizes meshes into 64×64×64 grids
- Creates occupancy grids (`occ.npz`)
- Assigns part labels (`part.npz`): 1=exterior, 2=interior, 3=roof, 4=floor
- Generates edit/protected masks for optimization
- Extracts FEA targets from CSVs
- Saves everything in `data/runs/{sample_id}/`

**Output structure per sample:**
```
data/runs/00000/
├── occ.npz              # Binary occupancy grid (64³)
├── part.npz             # Part labels (64³)
├── edit_mask.npz        # Regions allowed to be modified
├── protected_mask.npz   # Regions that must stay fixed
├── meta.json            # Materials, bounds, voxel size
└── targets.json         # FEA results (stress, displacement, etc.)
```

---

## Phase 3: ML Model Training

### 4. Build Dataset Index

```bash
python -m fea_ml.scripts.build_index \
    --runs-dir data/runs \
    --output data/manifests
```

**What it does:**
- Scans all samples in `data/runs/`
- Creates train/val/test splits (80/10/10)
- Saves manifests: `data/manifests/train.json`, `val.json`, `test.json`

---

### 5. Train Surrogate Model

```bash
python -m fea_ml.scripts.train \
    --config configs/voxel_config.yaml \
    --output runs/experiment1
```

**What it does:**
- Trains a 3D CNN to predict FEA metrics from voxel geometry
- Uses deep ensemble (5 models) for uncertainty quantification
- Trains for 100 epochs with mixed precision
- Saves best checkpoint to `runs/experiment1/best.pt`

**Training time estimate:**
- **Consumer GPU (RTX 3080/4090)**: ~10-20 hours for 50k samples
- **H100**: ~3-5 hours

**Monitor training:**
```bash
tensorboard --logdir runs/experiment1
```

---

### 6. Evaluate Model (Optional but Recommended)

```bash
python -m fea_ml.scripts.evaluate \
    --config configs/voxel_config.yaml \
    --checkpoint runs/experiment1/best.pt \
    --output runs/experiment1/eval
```

**What it does:**
- Tests model on held-out test set
- Generates prediction vs actual plots
- Computes R², MAE, RMSE for each target
- Validates uncertainty calibration

**Look for:**
- R² > 0.90 for all targets (good surrogate)
- Well-calibrated uncertainties (prediction intervals contain true values)

---

## Phase 4: Run Optimization 🎯

### 7. Optimize a Design

```bash
python -m fea_ml.scripts.optimize \
    --config configs/voxel_config.yaml \
    --checkpoint runs/experiment1/best.pt \
    --baseline data/runs/00000 \
    --output runs/experiment1/optimization/00000_optimized
```

**What it does:**
- Takes a baseline design (`data/runs/00000`)
- Uses CMA-ES to search for optimized geometry parameters
- Objectives:
  - **Minimize** material volume (lighter structure)
  - **Maintain** safety factor ≥ 1.5
  - **Maintain** displacement ≤ threshold
  - **Maintain** compliance ≤ 1.05× baseline
- Runs 50 iterations with 16 candidates per iteration

**Output:**
```
runs/experiment1/optimization/00000_optimized/
├── optimized_occ.npz       # Optimized voxel grid
├── optimized_mesh.stl      # Exportable STL
├── optimization_log.json   # Search history
└── final_metrics.json      # Performance comparison
```

**Expected results:**
- 10-30% volume reduction
- Structural constraints satisfied
- ~1-2 hours per design on consumer GPU

---

## Phase 5: Validation & Iteration

### 8. Verify Optimized Design (Critical!)

**Never trust the surrogate blindly!** You must validate:

```bash
# Extract optimized design to STEP/mesh
# Run actual FEA on optimized geometry
python solve_asce7_22_asd_sfepy_ai_labels.py --mesh optimized.msh
```

**Compare:**
- Predicted safety factor vs. actual
- If surrogate is accurate (within ±10%), design is valid
- If not, collect this as a training sample and retrain

### 9. Retrain with New Data

As you validate optimized designs:

```bash
# Add validated results to data/runs/
# Rebuild index
python -m fea_ml.scripts.build_index --runs-dir data/runs --output data/manifests

# Retrain (or finetune)
python -m fea_ml.scripts.train \
    --config configs/voxel_config.yaml \
    --resume runs/experiment1/best.pt \
    --output runs/experiment2
```

---

## Summary: Full Command Sequence

```bash
# Phase 1: Simulation
cd c:\Users\ericx\workspace\topopt_project\optimization\fea_gmsh_run
python gmsh_fragment_mesh.py --step-dir fea_out --output-dir fea_out --overwrite
python solve_asce7_22_asd_sfepy_ai_labels.py --mesh-dir fea_out --output-dir fea_out

# Phase 2: Data Prep (requires part extraction - see Option A/B above)
cd c:\Users\ericx\workspace\topopt_project\fea_ml
python -m fea_ml.scripts.prepare_real_data --parts-dir ../optimization/fea_gmsh_run/fea_out --fea-dir ../optimization/fea_gmsh_run/fea_out --output-dir data/runs

# Phase 3: Training
python -m fea_ml.scripts.build_index --runs-dir data/runs --output data/manifests
python -m fea_ml.scripts.train --config configs/voxel_config.yaml --output runs/experiment1
python -m fea_ml.scripts.evaluate --config configs/voxel_config.yaml --checkpoint runs/experiment1/best.pt --output runs/experiment1/eval

# Phase 4: Optimize
python -m fea_ml.scripts.optimize --config configs/voxel_config.yaml --checkpoint runs/experiment1/best.pt --baseline data/runs/00000 --output runs/experiment1/optimization/00000_opt

# Phase 5: Validate & Iterate
# (Manual FEA verification + retrain loop)
```

---

## Critical Bottleneck: Part Extraction

**You need to create a mesh-to-STL-parts extraction script** since your current workflow uses complete meshes with physical groups, but `prepare_real_data.py` expects separate part STLs.

Would you like me to create this extraction script?
