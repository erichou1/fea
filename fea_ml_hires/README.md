# FEA ML Surrogate — High-Resolution (512³–1024³)

Variant of the FEA-ML voxel surrogate pipeline specifically adapted for
**extreme-resolution** voxel grids (512³–1024³) on high-memory GPUs such as
the NVIDIA GB200 (192 GB HBM3e).

## Key Differences from `fea_ml` (128³)

| Aspect | `fea_ml` (128³) | `fea_ml_hires` (1024³) |
|--------|-----------------|------------------------|
| Input resolution | 128³ | 512³–1024³ |
| Stem downsampling | /4 (stride-2 conv + pool) | /8 (stride-4 conv + stride-2 pool) |
| Residual stages | 4 | 5 |
| Total spatial reduction | /64 | /256 |
| Base channels | 64 | 24 (memory-limited) |
| Channel progression | 64→128→256→512 | 24→48→96→192→384 |
| Batch size | 32 | 1–2 (gradient accumulation) |
| Gradient checkpointing | Optional | Mandatory |
| Precision | fp16 AMP | bf16 AMP (Hopper/Blackwell native) |
| torch.compile | Optional | Recommended |

## Memory Budget (1024³, bf16)

```
Input tensor   (7ch × 1024³ × 2B)     ≈ 14 GB
After stem     (24ch × 128³ × 2B)     ≈  100 MB
Stage 1        (48ch × 64³ × 2B)      ≈   25 MB
Stage 2        (96ch × 32³ × 2B)      ≈    6 MB
Stage 3+       (negligible)

Model params + optimizer states        ≈  200 MB
Peak working memory (w/ grad-ckpt)     ≈ ~40 GB
```

Fits comfortably on GB200 (192 GB).  Smaller GPUs can use 512³ resolution.

## Quick Start

### Installation

```bash
cd fea_ml_hires
pip install -r requirements.txt
```

### Train (single-GPU)

```bash
python -m fea_ml_hires.scripts.train \
    --config configs/hires_1024.yaml \
    --output runs/hires_1024_v1
```

### Train (multi-GPU with FSDP)

```bash
torchrun --nproc_per_node=4 -m fea_ml_hires.scripts.train \
    --config configs/hires_1024.yaml \
    --output runs/hires_1024_v1 \
    --fsdp
```

### Override Gradient Accumulation

```bash
python -m fea_ml_hires.scripts.train \
    --config configs/hires_1024.yaml \
    --output runs/hires_1024_v1 \
    --grad_accum 16
```

### Evaluate

```bash
python -m fea_ml_hires.scripts.evaluate \
    --config configs/hires_1024.yaml \
    --checkpoint runs/hires_1024_v1/best.pt \
    --output runs/hires_1024_v1/eval
```

### Optimize

```bash
python -m fea_ml_hires.scripts.optimize \
    --config configs/hires_1024.yaml \
    --checkpoint runs/hires_1024_v1/best.pt \
    --baseline data/runs_real_1024/sample_001 \
    --output runs/hires_1024_v1/optimization
```

## Data Format

Same as `fea_ml`, but all voxel grids are at the target resolution (e.g. 1024³):

| File | Description |
|------|-------------|
| `occ.npz` | Occupancy grid `(D,H,W)` uint8 `{0,1}` |
| `part.npz` | Part labels `(D,H,W)` uint8 `{0-5}` |
| `edit_mask.npz` | Editable regions `(D,H,W)` uint8 |
| `protected_mask.npz` | Protected regions `(D,H,W)` uint8 |
| `meta.json` | Material/load case metadata |
| `targets.json` | FEA simulation targets |

## Model Architecture

### `Surrogate3DResNet_HiRes`

```
Input (7ch, 1024³)
  ├─ Stem: Conv3d(7→24, k=7, s=4) + BN + GELU + MaxPool(k=3, s=2)  → (24, 128³)
  ├─ Stage 1: ResBlock(24→48, stride=2)      → (48, 64³)
  ├─ Stage 2: ResBlock(48→96, stride=2)      → (96, 32³)
  ├─ Stage 3: ResBlock(96→192, stride=2)     → (192, 16³)
  ├─ Stage 4: ResBlock(192→384, stride=2)    → (384, 8³)
  ├─ Stage 5: ResBlock(384→384, stride=2)    → (384, 4³)
  ├─ AdaptiveAvgPool → (384, 1³)
  ├─ Flatten → 384
  ├─ + Multi-scale pool features (768)
  ├─ + Global features (256)
  └─ Prediction head (1024→512→4 targets)
```

**Gradient checkpointing** is always enabled — forward activations for all
5 stages are recomputed during the backward pass to save memory.

### Ensemble

Default: 5-member Deep Ensemble (same as `fea_ml`).
Each member is trained independently with unique seeds, EMA averaging,
cosine-annealing LR, and early stopping.

### Uncertainty

Conservative constraint enforcement using `mean − k × std ≥ threshold`
with calibrated ensemble disagreement.

## Training Strategy

- **Batch size = 1** (each 1024³ sample ≈ 14 GB in bf16)
- **Gradient accumulation** (default 8 steps → effective batch 8)
- **bf16 autocast** (native Hopper/Blackwell support)
- **Gradient checkpointing** (recompute activations in backward)
- **torch.compile** (inductor backend for fused kernels)
- **EMA** (exponential moving average of weights)
- **Cosine-annealing LR** with 1% minimum

## Transfer Learning

The `cnn3d.py` module also includes the original `Surrogate3DResNet` (128³)
to support loading pretrained 128³ weights and initializing overlapping
layers of the high-res model.

## Configuration

See [configs/hires_1024.yaml](configs/hires_1024.yaml) for the complete
1024³ training configuration.  Key settings:

```yaml
model:
  backbone: resnet3d_hires
  base_channels: 24

training:
  batch_size: 1
  grad_accum_steps: 8
  compile: true
```

## GPU Requirements

| Resolution | GPU Memory | Recommended GPU |
|-----------|------------|-----------------|
| 512³ | ~80 GB | H100 80GB / A100 80GB |
| 1024³ | ~40–60 GB (w/ grad-ckpt) | GB200 192GB |
| 1024³ ensemble ×5 | Sequential training | GB200 192GB |

## Tests

```bash
pytest fea_ml_hires/fea_ml_hires/tests -v
```
