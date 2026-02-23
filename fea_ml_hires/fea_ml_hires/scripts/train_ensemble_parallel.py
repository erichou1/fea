"""
Parallel ensemble training across multiple GPUs.

Trains N ensemble members in parallel, distributing members across GPUs.
Each GPU independently trains its assigned members sequentially, but all
GPUs run concurrently — giving up to N_GPU× speedup.

Example (4 GPUs, 8 members → 2 members per GPU):
    python -m fea_ml_hires.scripts.train_ensemble_parallel \\
        --config fea_ml_hires/configs/hires_512.yaml \\
        --output runs/hires_512_v1 \\
        --n_gpus 4

All members share the same data splits and normalization stats (computed
once on CPU before forking).  Each member trains with a unique seed.

Monitoring:
    tensorboard --logdir runs/hires_512_v1/tensorboard
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader

from fea_ml_hires.data.voxel_dataset import (
    VoxelFEADataset,
    VoxelNormalizationStats,
    compute_voxel_normalization_stats,
    create_data_splits,
)
from fea_ml_hires.models.cnn3d import create_surrogate_model
from fea_ml_hires.utils.config import load_config
from fea_ml_hires.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_member_logger(
    member_id: int, gpu_id: int, log_dir: Path,
) -> logging.Logger:
    """Create a per-member logger that writes to its own file + stdout."""
    name = f"member_{member_id:02d}_gpu{gpu_id}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        f"%(asctime)s [M{member_id:02d}|GPU{gpu_id}] %(levelname)s - %(message)s"
    )

    # File handler
    fh = logging.FileHandler(log_dir / f"member_{member_id:02d}.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Stdout
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Per-member training (runs in a child process)
# ---------------------------------------------------------------------------
def train_member_on_gpu(
    member_id: int,
    gpu_id: int,
    config: Dict[str, Any],
    train_dirs: List[Path],
    val_dirs: List[Path],
    stats_dict: Dict[str, Any],
    output_dir: Path,
    base_seed: int,
) -> None:
    """Train a single ensemble member on a specific GPU.

    This function is the target for mp.Process.  It sets up its own
    CUDA device, data loaders, model, and training loop.
    """
    try:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)

        ensemble_dir = output_dir / "ensemble"
        ensemble_dir.mkdir(parents=True, exist_ok=True)
        log_dir = output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        logger = setup_member_logger(member_id, gpu_id, log_dir)
        logger.info(f"Starting training — GPU {gpu_id} "
                     f"({torch.cuda.get_device_name(gpu_id)}, "
                     f"{torch.cuda.get_device_properties(gpu_id).total_mem / 1e9:.1f} GB)")

        # Seed (unique per member)
        seed = base_seed + member_id * 1000
        set_seed(seed)
        logger.info(f"Seed: {seed}")

        # Check if already trained (resume support)
        checkpoint_path = ensemble_dir / f"ensemble_member_{member_id:02d}.pt"
        if checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location=device, weights_only=False)
            logger.info(f"ALREADY TRAINED — loaded from checkpoint "
                        f"(val_loss={state.get('val_loss', '?'):.4f}, "
                        f"epoch={state.get('epoch', '?')})")
            return

        # Reconstruct normalization stats
        stats = VoxelNormalizationStats.from_dict(stats_dict)

        # Create datasets
        target_names = tuple(config["targets"])
        material_types = tuple(config["materials"])
        load_cases = tuple(config["load_cases"])
        resolution = config["data"]["resolution"]
        use_sdf = config["data"].get("use_sdf", False)

        train_dataset = VoxelFEADataset(
            run_dirs=train_dirs,
            target_names=target_names,
            material_types=material_types,
            load_cases=load_cases,
            resolution=resolution,
            use_sdf=use_sdf,
            stats=stats,
            augment=True,
        )
        val_dataset = VoxelFEADataset(
            run_dirs=val_dirs,
            target_names=target_names,
            material_types=material_types,
            load_cases=load_cases,
            resolution=resolution,
            use_sdf=use_sdf,
            stats=stats,
            augment=False,
        )

        batch_size = config["training"]["batch_size"]
        nw = config["training"].get("num_workers", 4)
        if platform.system() == "Windows":
            nw = 0

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=nw, pin_memory=True,
            persistent_workers=nw > 0, prefetch_factor=2 if nw > 0 else None,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=nw, pin_memory=True,
            persistent_workers=nw > 0, prefetch_factor=2 if nw > 0 else None,
        )
        logger.info(f"Data: {len(train_dataset)} train, {len(val_dataset)} val, "
                     f"batch_size={batch_size}, num_workers={nw}")

        # Create model
        in_channels = train_dataset.get_voxel_channels()
        feature_dim = train_dataset.get_feature_dim()
        target_dim = len(target_names)

        model = create_surrogate_model(
            in_channels=in_channels,
            feature_dim=feature_dim,
            target_dim=target_dim,
            resolution=resolution,
            dropout=config["model"].get("dropout", 0.12),
            drop_path=config["model"].get("drop_path", 0.15),
            backbone=config["model"].get("backbone", "resnet3d_hires"),
            base_channels=config["model"].get("base_channels", 48),
        )
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model: {n_params:,} parameters")

        # torch.compile
        if config["training"].get("compile", False):
            try:
                model = torch.compile(model)
                logger.info("torch.compile enabled")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

        # Optimizer & scheduler with warmup
        epochs = config["training"]["epochs"]
        lr = config["training"]["lr"]
        weight_decay = config["training"].get("weight_decay", 1e-4)
        warmup_epochs = config["training"].get("warmup_epochs", 5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Cosine annealing with linear warmup
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=lr * 0.01,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
        logger.info(f"LR schedule: {warmup_epochs}-epoch warmup → cosine annealing")

        # Loss
        loss_fn = nn.SmoothL1Loss()
        grad_clip = config["training"].get("grad_clip", 1.0)
        grad_accum = config["training"].get("grad_accum_steps", 1)
        patience = config["training"].get("patience", 40)
        use_ema = config["training"].get("use_ema", True)
        ema_decay = config["training"].get("ema_decay", 0.9995)

        # Target weights
        target_weights = None
        if "target_weights" in config.get("loss", {}):
            weights = config["loss"]["target_weights"]
            weight_list = [weights.get(t, 1.0) for t in config["targets"]]
            target_weights = torch.tensor(weight_list, dtype=torch.float32, device=device)
            logger.info(f"Target weights: {dict(zip(config['targets'], weight_list))}")

        # TensorBoard writer (per-member)
        tb_dir = output_dir / "tensorboard" / f"member_{member_id:02d}"
        tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(str(tb_dir))
        except ImportError:
            logger.warning("TensorBoard not available")

        # AMP scaler
        use_amp = config["training"].get("mixed_precision", True) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # EMA state
        ema_state = copy.deepcopy(model.state_dict()) if use_ema else None

        # Monitoring config
        mon = config.get("monitoring", {})
        log_grad_norms = mon.get("log_grad_norms", True)
        log_weight_hist = mon.get("log_weight_histograms", False)
        log_memory = mon.get("log_memory", True)
        ckpt_every = mon.get("checkpoint_every", 10)
        early_warn_epoch = mon.get("early_warning_epoch", 10)

        # ---- Data sanity check (catch corrupted files before training) ----
        logger.info("Running data sanity check (first 5 samples)...")
        for si in range(min(5, len(train_dataset))):
            try:
                sample = train_dataset[si]
                v = sample["voxel"]
                t = sample["targets"]
                if torch.isnan(v).any() or torch.isinf(v).any():
                    logger.error(f"  Sanity check FAILED: sample {si} has NaN/Inf in voxel")
                    raise ValueError(f"Corrupt voxel data in sample {si}")
                if torch.isnan(t).any() or torch.isinf(t).any():
                    logger.error(f"  Sanity check FAILED: sample {si} has NaN/Inf in targets")
                    raise ValueError(f"Corrupt target data in sample {si}")
            except Exception as e:
                logger.error(f"  Sanity check FAILED on sample {si}: {e}")
                raise
        logger.info("  Data sanity check passed ✓")

        # Training loop
        best_val_loss = float("inf")
        wait = 0
        nan_count = 0  # Track consecutive NaN losses
        history = {"train_loss": [], "val_loss": [], "val_r2": [], "lr": [], "epoch_time": [],
                    "grad_norm": [], "gpu_mem_gb": []}

        # ---- Epoch-0 baseline validation (before any training) ----
        model.eval()
        baseline_preds, baseline_tgts = [], []
        with torch.no_grad():
            for batch in val_loader:
                voxel = batch["voxel"].to(device, non_blocking=True)
                features = batch["features"].to(device, non_blocking=True)
                targets_b = batch["targets"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    preds = model(voxel, features)
                baseline_preds.append(preds.cpu().numpy())
                baseline_tgts.append(targets_b.cpu().numpy())
        bp = np.concatenate(baseline_preds)
        bt = np.concatenate(baseline_tgts)
        baseline_r2 = []
        for i in range(bp.shape[1]):
            ss_res = ((bt[:, i] - bp[:, i]) ** 2).sum()
            ss_tot = ((bt[:, i] - bt[:, i].mean()) ** 2).sum()
            baseline_r2.append(float(1.0 - ss_res / (ss_tot + 1e-8)))
        logger.info(f"Baseline (untrained) R²: {[f'{r:.4f}' for r in baseline_r2]}")
        logger.info(f"  (Expected: strongly negative. If R²≈0, targets may be trivial.)")
        model.train()

        for epoch in range(epochs):
            t0 = time.time()

            # ------- TRAIN -------
            model.train()
            train_losses = []
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(train_loader):
                voxel = batch["voxel"].to(device, non_blocking=True)
                features = batch["features"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    preds = model(voxel, features)
                    if target_weights is not None:
                        per_t = ((preds - targets) ** 2).mean(dim=0)
                        loss = (per_t * target_weights).sum()
                    else:
                        loss = loss_fn(preds, targets)
                    loss = loss / grad_accum

                scaler.scale(loss).backward()

                if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    # EMA update
                    if ema_state is not None:
                        with torch.no_grad():
                            for k, v in model.state_dict().items():
                                if v.is_floating_point():
                                    ema_state[k].lerp_(v, 1.0 - ema_decay)
                                else:
                                    ema_state[k].copy_(v)

                raw_loss = loss.item() * grad_accum

                # NaN/Inf detection
                if not np.isfinite(raw_loss):
                    nan_count += 1
                    logger.warning(f"⚠️  NaN/Inf loss at epoch {epoch+1}, step {step+1} "
                                   f"(consecutive count: {nan_count})")
                    if nan_count >= 10:
                        logger.error("❌ 10 consecutive NaN/Inf losses — aborting training. "
                                     "Check data normalization, LR, or model architecture.")
                        raise RuntimeError("Training diverged (persistent NaN/Inf loss)")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                else:
                    nan_count = 0

                train_losses.append(raw_loss)

            scheduler.step()
            train_loss = float(np.mean(train_losses))

            # Gradient norm (for monitoring)
            grad_norm = 0.0
            if log_grad_norms:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = total_norm ** 0.5

            # ------- VALIDATE (with EMA weights) -------
            if ema_state is not None:
                real_state = copy.deepcopy(model.state_dict())
                model.load_state_dict(ema_state)

            model.eval()
            val_losses = []
            all_preds, all_tgts = [], []
            with torch.no_grad():
                for batch in val_loader:
                    voxel = batch["voxel"].to(device, non_blocking=True)
                    features = batch["features"].to(device, non_blocking=True)
                    targets_b = batch["targets"].to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        preds = model(voxel, features)
                        vloss = loss_fn(preds, targets_b)
                    val_losses.append(vloss.item())
                    all_preds.append(preds.cpu().numpy())
                    all_tgts.append(targets_b.cpu().numpy())

            val_loss = float(np.mean(val_losses))

            # Per-target R²
            p_arr = np.concatenate(all_preds)
            t_arr = np.concatenate(all_tgts)
            val_r2 = []
            for i in range(p_arr.shape[1]):
                ss_res = ((t_arr[:, i] - p_arr[:, i]) ** 2).sum()
                ss_tot = ((t_arr[:, i] - t_arr[:, i].mean()) ** 2).sum()
                val_r2.append(float(1.0 - ss_res / (ss_tot + 1e-8)))
            val_r2_arr = np.array(val_r2)

            # Per-target MAE and RMSE
            mae_per_target = np.abs(p_arr - t_arr).mean(axis=0)
            rmse_per_target = np.sqrt(((p_arr - t_arr) ** 2).mean(axis=0))

            # GPU memory
            gpu_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0
            torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None

            epoch_time = time.time() - t0

            # Restore real weights
            if ema_state is not None:
                model.load_state_dict(real_state)

            # --- Logging ---
            r2_strs = [f"{target_names[i]}={val_r2[i]:.4f}" for i in range(len(target_names))]
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"train={train_loss:.4f} val={val_loss:.4f} | "
                f"R²=[{', '.join(r2_strs)}] | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                f"grad_norm={grad_norm:.2f} | "
                f"mem={gpu_mem_gb:.1f}GB | "
                f"time={epoch_time:.1f}s"
            )

            # ---- Quality Checks ----

            # Early warning
            if epoch + 1 == early_warn_epoch:
                if all(r < 0 for r in val_r2):
                    logger.warning(
                        "⚠️  All R² values still negative! "
                        "Model may not be learning. Check data quality."
                    )
                elif all(r < 0.1 for r in val_r2):
                    logger.warning(
                        "⚠️  R² values very low. Learning is slow — "
                        "consider lowering LR or checking data."
                    )

            # Loss divergence detection: val_loss > 5× best
            if best_val_loss < float("inf") and val_loss > best_val_loss * 5.0:
                logger.warning(
                    f"⚠️  Val loss ({val_loss:.4f}) is >5× best ({best_val_loss:.4f}). "
                    f"Possible divergence!"
                )

            # Overfitting detection: train << val
            if epoch >= 20 and train_loss < val_loss * 0.3:
                logger.warning(
                    f"⚠️  Significant overfitting detected: "
                    f"train_loss={train_loss:.4f} << val_loss={val_loss:.4f} "
                    f"(ratio={train_loss/val_loss:.2f})"
                )

            # NaN in validation R²
            if any(not np.isfinite(r) for r in val_r2):
                logger.warning(
                    f"⚠️  Non-finite R² values detected: {val_r2}. "
                    f"Check target normalization."
                )

            # TensorBoard
            if tb_writer:
                tb_writer.add_scalar("Loss/train", train_loss, epoch)
                tb_writer.add_scalar("Loss/val", val_loss, epoch)
                tb_writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
                tb_writer.add_scalar("Training/grad_norm", grad_norm, epoch)
                tb_writer.add_scalar("Training/epoch_time_s", epoch_time, epoch)
                if log_memory:
                    tb_writer.add_scalar("System/gpu_mem_gb", gpu_mem_gb, epoch)
                for i, tn in enumerate(target_names):
                    tb_writer.add_scalar(f"R2/{tn}", val_r2[i], epoch)
                    tb_writer.add_scalar(f"MAE/{tn}", mae_per_target[i], epoch)
                    tb_writer.add_scalar(f"RMSE/{tn}", rmse_per_target[i], epoch)
                # Weight histograms (every N epochs, expensive)
                if log_weight_hist and (epoch + 1) % 10 == 0:
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            tb_writer.add_histogram(f"Weights/{name}", param.data.cpu(), epoch)
                tb_writer.flush()

            # History
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_r2"].append(val_r2)
            history["lr"].append(optimizer.param_groups[0]["lr"])
            history["epoch_time"].append(epoch_time)
            history["grad_norm"].append(grad_norm)
            history["gpu_mem_gb"].append(gpu_mem_gb)

            # Save best (EMA) checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                torch.save({
                    "model_state_dict": ema_state if ema_state is not None else model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_r2": val_r2,
                    "mae": mae_per_target.tolist(),
                    "rmse": rmse_per_target.tolist(),
                    "config": config,
                }, checkpoint_path)
                logger.info(f"  ★ New best (val_loss={val_loss:.4f})")
            else:
                wait += 1

            # Periodic checkpoint (for crash recovery)
            if ckpt_every > 0 and (epoch + 1) % ckpt_every == 0:
                periodic_path = ensemble_dir / f"member_{member_id:02d}_epoch{epoch+1}.pt"
                torch.save({
                    "model_state_dict": ema_state if ema_state is not None else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_r2": val_r2,
                }, periodic_path)

            # Early stopping
            if patience > 0 and wait >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} (patience={patience})")
                break

        # Save history
        with open(log_dir / f"member_{member_id:02d}_history.json", "w") as f:
            json.dump(history, f, indent=2)

        if tb_writer:
            tb_writer.close()

        logger.info(f"✓ Training complete — best val_loss={best_val_loss:.4f}")

    except Exception as e:
        print(f"\n{'='*60}", flush=True)
        print(f"MEMBER {member_id} (GPU {gpu_id}) CRASHED: {e}", flush=True)
        traceback.print_exc()
        print(f"{'='*60}", flush=True)
        raise


# ---------------------------------------------------------------------------
# Main: orchestrate parallel training
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train FEA surrogate ensemble in parallel across GPUs"
    )
    parser.add_argument("--config", type=str, required=True, help="Config YAML file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--n_gpus", type=int, default=None,
                        help="Number of GPUs to use (default: all available)")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to clean_manifest.json")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    import yaml
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # GPU detection
    n_available = torch.cuda.device_count()
    n_gpus = args.n_gpus or n_available
    if n_gpus > n_available:
        print(f"WARNING: Requested {n_gpus} GPUs but only {n_available} available. "
              f"Using {n_available}.")
        n_gpus = n_available
    if n_gpus == 0:
        raise RuntimeError("No GPUs available!")

    n_models = config["model"].get("n_models", 8)
    seed = config["training"].get("seed", 42)
    set_seed(seed)

    print(f"{'='*70}")
    print(f"  PARALLEL ENSEMBLE TRAINING")
    print(f"  {n_models} ensemble members across {n_gpus} GPUs")
    print(f"  Members per GPU: {(n_models + n_gpus - 1) // n_gpus}")
    print(f"  Config: {args.config}")
    print(f"  Output: {output_dir}")
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / 1e9
        print(f"  GPU {i}: {name} ({mem:.1f} GB)")
    print(f"{'='*70}")

    # ---- Data preparation (done once, shared across all members) ----
    runs_dir = Path(config["data"]["runs_dir"])
    train_dirs, val_dirs, test_dirs = create_data_splits(
        runs_dir,
        train_ratio=config["data"]["train_split"],
        val_ratio=config["data"]["val_split"],
        test_ratio=config["data"]["test_split"],
        seed=seed,
        split_by_family=config["data"].get("split_by_design_family", True),
    )

    # Filter by manifest
    if args.manifest:
        manifest_path = Path(args.manifest)
        with open(manifest_path) as f:
            manifest = json.load(f)
        clean_ids = set(manifest.get("clean_sample_ids", manifest if isinstance(manifest, list) else []))
        train_dirs = [d for d in train_dirs if d.name in clean_ids]
        val_dirs = [d for d in val_dirs if d.name in clean_ids]
        test_dirs = [d for d in test_dirs if d.name in clean_ids]
        print(f"  Manifest filter: {len(clean_ids)} clean IDs → "
              f"train={len(train_dirs)}, val={len(val_dirs)}, test={len(test_dirs)}")

    total = len(train_dirs) + len(val_dirs) + len(test_dirs)
    print(f"  Dataset: {len(train_dirs)} train, {len(val_dirs)} val, "
          f"{len(test_dirs)} test ({total} total)")

    if total < 100:
        print(f"  ⚠️  Very few samples ({total}). Expected ~14,000+.")
    if len(train_dirs) == 0:
        raise ValueError(f"No training samples in {runs_dir}")

    # Save splits
    with open(output_dir / "splits.json", "w") as f:
        json.dump({
            "train": [str(d) for d in train_dirs],
            "val": [str(d) for d in val_dirs],
            "test": [str(d) for d in test_dirs],
        }, f, indent=2)

    # Compute normalization stats (from training set)
    print("  Computing normalization stats...")
    target_names = tuple(config["targets"])
    material_types = tuple(config["materials"])
    load_cases = tuple(config["load_cases"])
    resolution = config["data"]["resolution"]
    use_sdf = config["data"].get("use_sdf", False)

    tmp_dataset = VoxelFEADataset(
        run_dirs=train_dirs,
        target_names=target_names,
        material_types=material_types,
        load_cases=load_cases,
        resolution=resolution,
        use_sdf=use_sdf,
        stats=None,
        augment=False,
    )
    log_transform = config.get("loss", {}).get("log_transform", None)
    stats = compute_voxel_normalization_stats(
        tmp_dataset,
        log_transform_targets=log_transform,
        winsorize_percentile=config.get("loss", {}).get("winsorize_pct", 2.0),
    )
    stats_dict = stats.to_dict()
    del tmp_dataset

    with open(output_dir / "normalization.json", "w") as f:
        json.dump(stats_dict, f, indent=2)
    print("  Normalization stats saved.")

    # Log model info
    sample_model = create_surrogate_model(
        in_channels=7, feature_dim=8, target_dim=len(target_names),
        resolution=resolution,
        dropout=config["model"].get("dropout", 0.12),
        drop_path=config["model"].get("drop_path", 0.15),
        backbone=config["model"].get("backbone", "resnet3d_hires"),
        base_channels=config["model"].get("base_channels", 48),
    )
    n_params = sum(p.numel() for p in sample_model.parameters())
    print(f"  Model: {n_params:,} params "
          f"(backbone={config['model'].get('backbone')}, "
          f"base_channels={config['model'].get('base_channels')})")
    del sample_model

    # Estimate training time
    batch_size = config["training"]["batch_size"]
    n_train = len(train_dirs)
    iters_per_epoch = (n_train + batch_size - 1) // batch_size
    epochs = config["training"]["epochs"]
    # Rough estimate: ~0.5-2s per iteration at 512³ bs=4 on GB200
    est_sec_per_iter = 1.0
    est_epoch_sec = iters_per_epoch * est_sec_per_iter
    members_per_gpu = (n_models + n_gpus - 1) // n_gpus
    est_total_hours = (est_epoch_sec * epochs * members_per_gpu) / 3600
    print(f"\n  ⏱  Estimated training time:")
    print(f"     {iters_per_epoch} iters/epoch × {epochs} epochs × "
          f"{members_per_gpu} members/GPU")
    print(f"     ≈ {est_total_hours:.1f} hours (with early stopping likely ~{est_total_hours*0.5:.1f}h)")

    # ---- Distribute members across GPUs ----
    gpu_assignments: Dict[int, List[int]] = {g: [] for g in range(n_gpus)}
    for m in range(n_models):
        gpu_assignments[m % n_gpus].append(m)

    print(f"\n  GPU assignments:")
    for g, members in gpu_assignments.items():
        print(f"    GPU {g}: members {members}")

    # ---- Launch parallel processes ----
    print(f"\n  Launching {n_gpus} worker processes...")
    mp.set_start_method("spawn", force=True)

    processes = []
    for gpu_id in range(n_gpus):
        member_ids = gpu_assignments[gpu_id]
        for member_id in member_ids:
            p = mp.Process(
                target=_train_wrapper,
                args=(member_id, gpu_id, config, train_dirs, val_dirs,
                      stats_dict, output_dir, seed),
                daemon=False,
            )
            # We start all first-round members (one per GPU) in parallel,
            # then wait for each GPU's sequential members
            processes.append((gpu_id, member_id, p))

    # Start all processes for the first member of each GPU
    active: Dict[int, Tuple[int, mp.Process]] = {}
    pending: Dict[int, List[Tuple[int, mp.Process]]] = {g: [] for g in range(n_gpus)}

    for gpu_id, member_id, p in processes:
        if gpu_id not in active:
            active[gpu_id] = (member_id, p)
            p.start()
            print(f"  Started member {member_id} on GPU {gpu_id} (PID {p.pid})")
        else:
            pending[gpu_id].append((member_id, p))

    # Wait and launch next member on each GPU as they finish
    completed = 0
    failed = []
    while active:
        for gpu_id in list(active.keys()):
            member_id, proc = active[gpu_id]
            proc.join(timeout=5.0)  # poll every 5s
            if not proc.is_alive():
                if proc.exitcode == 0:
                    completed += 1
                    print(f"  ✓ Member {member_id} (GPU {gpu_id}) completed "
                          f"[{completed}/{n_models}]")
                else:
                    failed.append(member_id)
                    print(f"  ✗ Member {member_id} (GPU {gpu_id}) FAILED "
                          f"(exit code {proc.exitcode})")

                # Launch next member for this GPU
                if pending[gpu_id]:
                    next_mid, next_p = pending[gpu_id].pop(0)
                    active[gpu_id] = (next_mid, next_p)
                    next_p.start()
                    print(f"  Started member {next_mid} on GPU {gpu_id} "
                          f"(PID {next_p.pid})")
                else:
                    del active[gpu_id]

    # ---- Summary ----
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE")
    print(f"  Successful: {completed}/{n_models}")
    if failed:
        print(f"  Failed members: {failed}")
    print(f"  Output: {output_dir}")
    print(f"  TensorBoard: tensorboard --logdir {output_dir/'tensorboard'}")

    # Collect all member results into summary
    ensemble_dir = output_dir / "ensemble"
    summary = []
    for m in range(n_models):
        cp = ensemble_dir / f"ensemble_member_{m:02d}.pt"
        if cp.exists():
            state = torch.load(cp, map_location="cpu", weights_only=False)
            summary.append({
                "member": m,
                "epoch": state.get("epoch", -1),
                "val_loss": state.get("val_loss", -1),
                "val_r2": state.get("val_r2", []),
                "mae": state.get("mae", []),
                "rmse": state.get("rmse", []),
            })
            r2_str = ", ".join(f"{target_names[i]}={state['val_r2'][i]:.4f}"
                               for i in range(len(target_names))
                               if i < len(state.get("val_r2", [])))
            print(f"  Member {m}: val_loss={state['val_loss']:.4f}, R²=[{r2_str}]")
        else:
            print(f"  Member {m}: NO CHECKPOINT (training failed)")

    with open(output_dir / "ensemble_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{'='*70}")

    if failed:
        sys.exit(1)


def _train_wrapper(member_id, gpu_id, config, train_dirs, val_dirs,
                   stats_dict, output_dir, base_seed):
    """Wrapper that ensures exceptions propagate cleanly to the parent."""
    try:
        train_member_on_gpu(
            member_id=member_id,
            gpu_id=gpu_id,
            config=config,
            train_dirs=train_dirs,
            val_dirs=val_dirs,
            stats_dict=stats_dict,
            output_dir=output_dir,
            base_seed=base_seed,
        )
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
