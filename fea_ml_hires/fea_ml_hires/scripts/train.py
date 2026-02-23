"""
Training script for HIGH-RESOLUTION voxel-based FEA surrogate (512^3 / 1024^3).

Designed for NVIDIA GB200 (192 GB HBM3e).  Key adaptations over the 128^3 version:
  - bf16 mixed precision throughout (via torch.amp.autocast)
  - Gradient accumulation to simulate larger effective batch sizes
    with batch_size=1-2 per GPU
  - torch.compile support (PyTorch 2.0+, highly recommended)
  - Gradient checkpointing (enabled in the model itself)
  - Memory-efficient data loading for multi-GB voxel grids
  - FSDP / DDP multi-GPU support (optional, via --fsdp flag)

Usage:
    # Single GPU (GB200, 192 GB)
    python -m fea_ml_hires.scripts.train \\
        --config configs/hires_1024.yaml --output runs/hires_v1

    # Multi-GPU with FSDP
    torchrun --nproc_per_node=4 -m fea_ml_hires.scripts.train \\
        --config configs/hires_1024.yaml --output runs/hires_v1 --fsdp
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from fea_ml_hires.data.voxel_dataset import (
    VoxelFEADataset,
    VoxelNormalizationStats,
    compute_voxel_normalization_stats,
    create_data_splits,
)
from fea_ml_hires.models.cnn3d import create_surrogate_model
from fea_ml_hires.models.ensemble import train_deep_ensemble
from fea_ml_hires.utils.config import load_config
from fea_ml_hires.utils.seed import set_seed


def _num_workers() -> int:
    """Return optimal DataLoader workers for current OS."""
    if platform.system() == "Windows":
        return 0  # Windows multiprocessing can hang
    import os
    # Cap at 4 to avoid fork/memory issues on large machines
    return min(os.cpu_count() or 4, 4)


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler — flush after every line so crashes don't lose messages
    fh = logging.FileHandler(output_dir / "train.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Also redirect stderr to the log file so crashes are captured
    import sys
    log_file = open(output_dir / "train_stderr.log", "w")
    sys.stderr = log_file
    
    return logger


def create_loss_fn(config: Dict) -> nn.Module:
    """Create loss function based on config."""
    loss_type = config.get("loss", {}).get("type", "huber")
    
    if loss_type == "huber":
        return nn.SmoothL1Loss()
    elif loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "mae":
        return nn.L1Loss()
    elif loss_type == "log_cosh":
        return LogCoshLoss()
    else:
        return nn.SmoothL1Loss()


class LogCoshLoss(nn.Module):
    """Log-Cosh loss: smoother than Huber, good for regression."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(diff + torch.nn.functional.softplus(-2.0 * diff) - 0.6931472)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: Optional[object],
    target_weights: Optional[torch.Tensor],
    grad_clip: float = 1.0,
    scheduler_onecycle: Optional[object] = None,
    grad_accum_steps: int = 1,
) -> float:
    """Train for one epoch with gradient accumulation, bf16, and clipping.

    Gradient accumulation allows effective batch_size = real_batch_size * grad_accum_steps
    without increasing peak GPU memory.  Essential for 1024^3 where batch_size=1.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc="Training", leave=False)
    for step, batch in enumerate(pbar):
        voxel = batch["voxel"].to(device, non_blocking=True)
        features = batch["features"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        if scaler is not None:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                predictions = model(voxel, features)
                if target_weights is not None:
                    per_target = ((predictions - targets) ** 2).mean(dim=0)
                    loss = (per_target * target_weights).sum()
                else:
                    loss = loss_fn(predictions, targets)
                loss = loss / grad_accum_steps  # scale for accumulation

            scaler.scale(loss).backward()
        else:
            predictions = model(voxel, features)
            if target_weights is not None:
                per_target_loss = ((predictions - targets) ** 2).mean(dim=0)
                loss = (per_target_loss * target_weights).sum()
            else:
                loss = loss_fn(predictions, targets)
            loss = loss / grad_accum_steps
            loss.backward()

        # Accumulate gradients, then step
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if scheduler_onecycle is not None:
                scheduler_onecycle.step()

        total_loss += loss.item() * grad_accum_steps  # un-scale for logging
        n_batches += 1
        pbar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})

    return total_loss / max(n_batches, 1)


def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    n_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for batch in pbar:
            voxel = batch["voxel"].to(device)
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            
            predictions = model(voxel, features)
            loss = loss_fn(predictions, targets)
            
            total_loss += loss.item()
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Per-target R² (in normalised space)
    r2_per_target = []
    for i in range(preds.shape[1]):
        ss_res = ((targets[:, i] - preds[:, i]) ** 2).sum()
        ss_tot = ((targets[:, i] - targets[:, i].mean()) ** 2).sum()
        r2_per_target.append(1.0 - ss_res / (ss_tot + 1e-8))

    mae = np.abs(preds - targets).mean(axis=0)
    rmse = np.sqrt(((preds - targets) ** 2).mean(axis=0))
    
    return {
        "loss": total_loss / max(n_batches, 1),
        "mae": mae,
        "rmse": rmse,
        "r2": np.array(r2_per_target),
    }


def train_single_model(
    config: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    output_dir: Path,
    device: torch.device,
    logger: logging.Logger,
) -> nn.Module:
    """Train a single model with all bells and whistles."""
    epochs = config["training"]["epochs"]
    lr = config["training"]["lr"]
    weight_decay = config["training"].get("weight_decay", 1e-4)
    use_amp = config["training"].get("mixed_precision", True)
    grad_clip = config["training"].get("grad_clip", 1.0)
    patience = config["training"].get("patience", 30)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Scheduler: CosineAnnealing with warm restart
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(epochs // 3, 10), T_mult=2, eta_min=lr * 0.01
    )
    
    loss_fn = create_loss_fn(config)
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None
    
    # Target weights
    target_weights = None
    if "target_weights" in config.get("loss", {}):
        weights = config["loss"]["target_weights"]
        weight_list = [weights.get(t, 1.0) for t in config["targets"]]
        target_weights = torch.tensor(weight_list, device=device)
    
    # TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(output_dir / "tensorboard")
    except ImportError:
        writer = None
    
    # Training loop
    best_val_loss = float("inf")
    metrics_history = []
    wait = 0
    
    for epoch in range(epochs):
        start_time = time.time()
        
        grad_accum = config["training"].get("grad_accum_steps", 1)
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler,
            target_weights, grad_clip=grad_clip, grad_accum_steps=grad_accum,
        )
        val_metrics = validate(model, val_loader, loss_fn, device)
        
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # Log
        r2_str = ", ".join(f"{r:.3f}" for r in val_metrics["r2"])
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"train: {train_loss:.4f}, val: {val_metrics['loss']:.4f}, "
            f"R²=[{r2_str}], lr: {optimizer.param_groups[0]['lr']:.2e}, "
            f"time: {epoch_time:.1f}s"
        )
        
        # TensorBoard
        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
            for i, target_name in enumerate(config["targets"]):
                writer.add_scalar(f"MAE/{target_name}", val_metrics["mae"][i], epoch)
                writer.add_scalar(f"R2/{target_name}", val_metrics["r2"][i], epoch)
        
        # Save metrics
        metrics_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "mae": val_metrics["mae"].tolist(),
            "rmse": val_metrics["rmse"].tolist(),
            "r2": val_metrics["r2"].tolist(),
        })
        
        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            wait = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": config,
            }, output_dir / "best.pt")
            logger.info(f"  ** New best model (val_loss: {best_val_loss:.4f})")
        else:
            wait += 1
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, output_dir / f"checkpoint_epoch{epoch+1}.pt")
        
        # Early stopping
        if patience > 0 and wait >= patience:
            logger.info(f"Early stopping at epoch {epoch+1} (patience={patience})")
            break
    
    if writer:
        writer.close()
    
    # Save metrics CSV
    import csv
    with open(output_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "mae", "rmse", "r2"])
        w.writeheader()
        w.writerows(metrics_history)
    
    # Load best model
    best_state = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state_dict"])
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train high-res FEA voxel surrogate")
    parser.add_argument("--config", type=str, required=True, help="Config YAML file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to clean_manifest.json — only train on listed sample IDs")
    parser.add_argument("--grad_accum", type=int, default=None,
                        help="Override gradient-accumulation steps (default: from config)")
    parser.add_argument("--fsdp", action="store_true",
                        help="Enable FSDP via torchrun (set automatically when LOCAL_RANK is set)")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # CLI overrides ----------------------------------------------------------
    if args.grad_accum is not None:
        config.setdefault("training", {})["grad_accum_steps"] = args.grad_accum
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        import yaml
        yaml.dump(config, f)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting training with config: {args.config}")
    
    # Set seed
    seed = config["training"].get("seed", 42)
    set_seed(seed)
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Log GPU info for debugging
    if device.type == "cuda":
        n_gpus = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        logger.info(f"Using device: {device}  ({n_gpus} GPUs: {gpu_name}, {gpu_mem:.1f}GB each)")
    else:
        logger.info(f"Using device: {device}")
    
    # Create data splits
    runs_dir = Path(config["data"]["runs_dir"])
    train_dirs, val_dirs, test_dirs = create_data_splits(
        runs_dir,
        train_ratio=config["data"]["train_split"],
        val_ratio=config["data"]["val_split"],
        test_ratio=config["data"]["test_split"],
        seed=seed,
        split_by_family=config["data"].get("split_by_design_family", True),
    )

    # --- Filter by clean manifest if provided ---
    if args.manifest:
        manifest_path = Path(args.manifest)
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        clean_ids = set(manifest.get("clean_sample_ids", manifest if isinstance(manifest, list) else []))
        logger.info(f"Loaded manifest with {len(clean_ids)} clean sample IDs from {manifest_path}")

        def _filter(dirs):
            return [d for d in dirs if d.name in clean_ids]

        before = (len(train_dirs), len(val_dirs), len(test_dirs))
        train_dirs = _filter(train_dirs)
        val_dirs = _filter(val_dirs)
        test_dirs = _filter(test_dirs)
        after = (len(train_dirs), len(val_dirs), len(test_dirs))
        logger.info(f"Manifest filtering: train {before[0]}→{after[0]}, "
                     f"val {before[1]}→{after[1]}, test {before[2]}→{after[2]}")
    
    
    total_samples = len(train_dirs) + len(val_dirs) + len(test_dirs)
    logger.info(f"Data splits: train={len(train_dirs)}, val={len(val_dirs)}, test={len(test_dirs)} (total={total_samples})")
    
    if total_samples < 100:
        logger.warning(f"*** VERY FEW SAMPLES ({total_samples}). "
                       f"Expected ~14000. Check runs_dir: {runs_dir.resolve()} ***")
        logger.warning("Training may fail or produce poor results with < 100 samples.")
    if len(train_dirs) == 0:
        raise ValueError(f"No training samples found in {runs_dir}. Is the dataset generated?")
    if len(val_dirs) == 0:
        raise ValueError(f"No validation samples found. Need more data in {runs_dir}.")
    
    # Save splits
    with open(output_dir / "splits.json", "w") as f:
        json.dump({
            "train": [str(d) for d in train_dirs],
            "val": [str(d) for d in val_dirs],
            "test": [str(d) for d in test_dirs],
        }, f, indent=2)
    
    # Create datasets (without normalization first to compute stats)
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
        stats=None,
        augment=True,
    )
    
    # Compute normalization stats (log transforms ALL targets by default)
    log_transform = config.get("loss", {}).get("log_transform", None)  # None = all
    stats = compute_voxel_normalization_stats(
        train_dataset,
        log_transform_targets=log_transform,
        winsorize_percentile=config.get("loss", {}).get("winsorize_pct", 2.0),
    )
    
    # Save normalization stats
    with open(output_dir / "normalization.json", "w") as f:
        json.dump(stats.to_dict(), f, indent=2)
    
    # Apply stats to datasets
    train_dataset.stats = stats
    
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
    
    # Create data loaders
    batch_size = config["training"]["batch_size"]
    nw = _num_workers()
    logger.info(f"DataLoader num_workers: {nw}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=nw > 0,
        prefetch_factor=2 if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=nw > 0,
        prefetch_factor=2 if nw > 0 else None,
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    
    # Create model(s)
    in_channels = train_dataset.get_voxel_channels()
    feature_dim = train_dataset.get_feature_dim()
    target_dim = len(target_names)
    dropout = config["model"].get("dropout", 0.15)
    drop_path = config["model"].get("drop_path", 0.1)
    backbone = config["model"].get("backbone", "cnn3d")
    base_channels = config["model"].get("base_channels", None)
    
    def model_factory():
        return create_surrogate_model(
            in_channels=in_channels,
            feature_dim=feature_dim,
            target_dim=target_dim,
            resolution=resolution,
            dropout=dropout,
            drop_path=drop_path,
            backbone=backbone,
            base_channels=base_channels,
        )
    
    # Log model details
    sample_model = model_factory()
    n_params = sum(p.numel() for p in sample_model.parameters())
    logger.info(f"Model: {backbone}, params: {n_params:,}, base_channels: {base_channels}, "
                f"dropout: {dropout}, drop_path: {drop_path}")
    logger.info(f"Normalization: log_transform={stats.log_transform_targets}, "
                f"target_mean(log)={stats.target_mean.tolist()}, "
                f"target_std(log)={stats.target_std.tolist()}")
    del sample_model
    
    # Train
    model_type = config["model"].get("type", "single")
    
    if model_type == "ensemble":
        n_models = config["model"].get("n_models", 5)
        logger.info(f"Training ensemble with {n_models} models")
        
        # Build target_weights tensor for ensemble training
        target_weights_tensor = None
        if "target_weights" in config.get("loss", {}):
            weights = config["loss"]["target_weights"]
            weight_list = [weights.get(t, 1.0) for t in config["targets"]]
            target_weights_tensor = torch.tensor(weight_list, dtype=torch.float32)
            logger.info(f"Target weights: {dict(zip(config['targets'], weight_list))}")

        try:
            from fea_ml_hires.models.ensemble import train_deep_ensemble
            ensemble = train_deep_ensemble(
                model_factory=model_factory,
                train_loader=train_loader,
                val_loader=val_loader,
                n_models=n_models,
                epochs=config["training"]["epochs"],
                lr=config["training"]["lr"],
                weight_decay=config["training"].get("weight_decay", 1e-4),
                device=device,
                output_dir=output_dir / "ensemble",
                base_seed=seed,
                grad_clip=config["training"].get("grad_clip", 1.0),
                patience=config["training"].get("patience", 20),
                use_ema=config["training"].get("use_ema", True),
                ema_decay=config["training"].get("ema_decay", 0.999),
                target_weights=target_weights_tensor,
                target_names=list(config["targets"]),
                grad_accum_steps=config["training"].get("grad_accum_steps", 1),
            )
            
            # Save ensemble checkpoint paths (best checkpoints already saved during training)
            ensemble_dir = output_dir / "ensemble"
            checkpoint_paths = sorted(ensemble_dir.glob("ensemble_member_*.pt"))
            with open(output_dir / "ensemble_paths.json", "w") as f:
                json.dump([str(p) for p in checkpoint_paths], f, indent=2)
            
            logger.info("Ensemble training complete")
        except Exception as e:
            import traceback
            logger.error(f"Ensemble training FAILED: {e}")
            logger.error(traceback.format_exc())
            raise
        
    else:
        model = model_factory().to(device)
        
        # Optional torch.compile (PyTorch 2.0+)
        if config["training"].get("compile", False):
            try:
                model = torch.compile(model)
                logger.info("torch.compile enabled")
            except Exception as e:
                logger.warning(f"torch.compile failed, using eager mode: {e}")
        
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        try:
            model = train_single_model(
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                model=model,
                output_dir=output_dir,
                device=device,
                logger=logger,
            )
            logger.info("Training complete")
        except Exception as e:
            import traceback
            logger.error(f"Training FAILED: {e}")
            logger.error(traceback.format_exc())
            raise


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Last-resort crash capture — writes to stdout even if logger is broken
        print(f"\n{'='*60}", flush=True)
        print(f"FATAL ERROR: {e}", flush=True)
        traceback.print_exc()
        print(f"{'='*60}", flush=True)
        sys.exit(1)
