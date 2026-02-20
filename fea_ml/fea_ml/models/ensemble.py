"""
Deep Ensemble for uncertainty estimation in FEA surrogate predictions (v2).

Improvements:
  - Cosine-annealing LR scheduler per member
  - Gradient clipping (max_norm)
  - Early stopping with configurable patience
  - Exponential Moving Average (EMA) weights
  - Per-target loss weighting option
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn


@dataclass
class EnsemblePrediction:
    """Prediction from an ensemble model."""
    mean: np.ndarray  # (B, T) mean prediction
    std: np.ndarray   # (B, T) standard deviation (uncertainty)
    predictions: np.ndarray  # (B, N, T) individual model predictions


class DeepEnsemble(nn.Module):
    """
    Deep Ensemble wrapper for uncertainty quantification.
    
    Manages N independent models and computes mean/std predictions.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            models: List of N independent surrogate models
            device: Device to move models to
        """
        super().__init__()
        
        self.n_models = len(models)
        self.models = nn.ModuleList(models)
        
        if device is not None:
            self.to(device)
    
    def forward(
        self,
        voxel: torch.Tensor,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through all ensemble members.
        
        Args:
            voxel: (B, C, D, H, W) voxel input
            features: (B, F) non-spatial features
            
        Returns:
            (mean, std) tensors, each (B, T)
        """
        predictions = []
        
        for model in self.models:
            pred = model(voxel, features)
            predictions.append(pred)
        
        # Stack predictions: (N, B, T)
        stacked = torch.stack(predictions, dim=0)
        
        # Compute mean and std
        mean = stacked.mean(dim=0)  # (B, T)
        std = stacked.std(dim=0)    # (B, T)
        
        return mean, std
    
    def predict_with_uncertainty(
        self,
        voxel: torch.Tensor,
        features: torch.Tensor,
        return_individual: bool = False,
    ) -> EnsemblePrediction:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            voxel: (B, C, D, H, W) voxel input
            features: (B, F) non-spatial features
            return_individual: Whether to return individual predictions
            
        Returns:
            EnsemblePrediction with mean, std, and optionally individual predictions
        """
        self.eval()
        
        with torch.no_grad():
            predictions = []
            
            for model in self.models:
                pred = model(voxel, features)
                predictions.append(pred.cpu().numpy())
            
            # Stack: (N, B, T)
            stacked = np.stack(predictions, axis=0)
            
            # Compute stats
            mean = stacked.mean(axis=0)  # (B, T)
            std = stacked.std(axis=0)    # (B, T)
            
            if return_individual:
                individual = stacked.transpose(1, 0, 2)  # (B, N, T)
            else:
                individual = np.empty((0,))
        
        return EnsemblePrediction(
            mean=mean,
            std=std,
            predictions=individual,
        )
    
    def enable_mc_dropout(self, mc_samples: int = 1) -> None:
        """Enable MC Dropout in all models (for combined uncertainty)."""
        for model in self.models:
            if hasattr(model, "enable_mc_dropout"):
                model.enable_mc_dropout()
    
    def disable_mc_dropout(self) -> None:
        """Disable MC Dropout."""
        for model in self.models:
            if hasattr(model, "disable_mc_dropout"):
                model.disable_mc_dropout()
    
    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_paths: List[Path],
        model_factory: callable,
        device: Optional[torch.device] = None,
    ) -> "DeepEnsemble":
        """
        Load ensemble from multiple checkpoint files.
        
        Args:
            checkpoint_paths: List of paths to model checkpoints
            model_factory: Function that creates a new model instance
            device: Device to load models to
            
        Returns:
            DeepEnsemble instance
        """
        models = []
        
        for path in checkpoint_paths:
            model = model_factory()
            state_dict = torch.load(path, map_location="cpu", weights_only=False)
            
            # Handle different checkpoint formats
            if "model_state_dict" in state_dict:
                model.load_state_dict(state_dict["model_state_dict"])
            elif "state_dict" in state_dict:
                model.load_state_dict(state_dict["state_dict"])
            else:
                model.load_state_dict(state_dict)
            
            models.append(model)
        
        return cls(models, device=device)
    
    def save_checkpoints(
        self,
        output_dir: Path,
        prefix: str = "ensemble_member",
        extra_state: Optional[Dict] = None,
    ) -> List[Path]:
        """
        Save all ensemble members as separate checkpoints.
        
        Args:
            output_dir: Directory to save checkpoints
            prefix: Filename prefix
            extra_state: Additional state to save with each checkpoint
            
        Returns:
            List of checkpoint paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = []
        for i, model in enumerate(self.models):
            path = output_dir / f"{prefix}_{i:02d}.pt"
            
            state = {
                "model_state_dict": model.state_dict(),
                "ensemble_index": i,
            }
            if extra_state:
                state.update(extra_state)
            
            torch.save(state, path)
            paths.append(path)
        
        return paths


def _weighted_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    base_loss_fn: nn.Module,
    target_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute loss with optional per-target weighting.

    When *target_weights* is not None the loss is:
        sum_i  w_i * SmoothL1Loss(pred_i, tgt_i)
    so that safety-critical targets can be emphasised.
    """
    if target_weights is None:
        return base_loss_fn(predictions, targets)

    # Per-target loss (reduce over batch, keep target dim)
    per_target = torch.stack([
        base_loss_fn(predictions[:, i], targets[:, i])
        for i in range(predictions.shape[1])
    ])
    return (per_target * target_weights).sum()


def _compute_val_r2(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_targets: int,
) -> np.ndarray:
    """Compute per-target R² on validation set (in normalised space).

    Returns an array of shape ``(n_targets,)`` with one R² value each.
    """
    all_preds: List[np.ndarray] = []
    all_tgts: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            voxel = batch["voxel"].to(device)
            features = batch["features"].to(device)
            targets = batch["targets"]
            preds = model(voxel, features).cpu().numpy()
            all_preds.append(preds)
            all_tgts.append(targets.numpy())

    preds = np.concatenate(all_preds, axis=0)
    tgts = np.concatenate(all_tgts, axis=0)

    r2 = np.zeros(n_targets)
    for i in range(n_targets):
        ss_res = ((tgts[:, i] - preds[:, i]) ** 2).sum()
        ss_tot = ((tgts[:, i] - tgts[:, i].mean()) ** 2).sum()
        r2[i] = 1.0 - ss_res / (ss_tot + 1e-8)
    return r2


def train_ensemble_member(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device,
    seed: int,
    checkpoint_path: Optional[Path] = None,
    scheduler: Optional[object] = None,
    grad_clip: float = 1.0,
    patience: int = 20,
    use_ema: bool = True,
    ema_decay: float = 0.999,
    target_weights: Optional[torch.Tensor] = None,
    target_names: Optional[List[str]] = None,
) -> Dict[str, List]:
    """
    Train a single ensemble member with all modern best-practices.

    Features:
      - Cosine-annealing LR (passed in as scheduler)
      - Gradient clipping
      - Early stopping with patience
      - Exponential Moving Average (EMA) of weights
      - Per-target loss weighting via *target_weights*
      - Per-target R² monitoring every epoch (logged for training validation)
    """
    import random
    from tqdm import tqdm

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = model.to(device)
    n_targets = len(target_names) if target_names else 4
    history: Dict[str, List] = {
        "train_loss": [],
        "val_loss": [],
        "val_r2": [],           # list of arrays, one per epoch
    }
    best_val_loss = float("inf")
    wait = 0  # patience counter

    # Move target_weights to device if provided
    if target_weights is not None:
        target_weights = target_weights.to(device)

    # Use per-target reduction when weights are provided
    base_loss_fn = nn.SmoothL1Loss(reduction="none" if target_weights is not None else "mean")

    # EMA model
    ema_state = None
    if use_ema:
        ema_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{epochs} [train]", leave=False)
        for batch in pbar:
            voxel = batch["voxel"].to(device)
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(voxel, features)
            loss = _weighted_loss(predictions, targets, loss_fn, target_weights)
            loss.backward()

            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            # EMA update
            if ema_state is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if v.is_floating_point():
                            ema_state[k].lerp_(v, 1.0 - ema_decay)
                        else:
                            ema_state[k].copy_(v)  # int tensors (e.g. num_batches_tracked)

            train_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # LR scheduler step
        if scheduler is not None:
            scheduler.step()

        # --- Validation (use EMA weights if available) ---
        if ema_state is not None:
            real_state = copy.deepcopy(model.state_dict())
            model.load_state_dict(ema_state)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                voxel = batch["voxel"].to(device)
                features = batch["features"].to(device)
                targets = batch["targets"].to(device)
                predictions = model(voxel, features)
                loss = _weighted_loss(predictions, targets, loss_fn, target_weights)
                val_losses.append(loss.item())

        # Per-target R² (on EMA model)
        val_r2 = _compute_val_r2(model, val_loader, device, n_targets)

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_r2"].append(val_r2.tolist())

        lr_str = ""
        if scheduler is not None:
            lr_str = f", lr: {scheduler.get_last_lr()[0]:.2e}"

        # Format R² per target
        if target_names:
            r2_parts = [f"{name}={r:.3f}" for name, r in zip(target_names, val_r2)]
        else:
            r2_parts = [f"t{i}={r:.3f}" for i, r in enumerate(val_r2)]
        r2_str = ", ".join(r2_parts)

        print(f"  Epoch {epoch+1}/{epochs} - train: {train_loss:.4f}, val: {val_loss:.4f}{lr_str}")
        print(f"    R²: [{r2_str}]")

        # Early warning: if R² is still all-negative after 10 epochs, something is wrong
        if epoch == 9:
            if all(r < 0 for r in val_r2):
                print(f"  WARNING: All R^2 values negative after 10 epochs! "
                      f"Model may not be learning -- check data quality / features.")

        # Save best (EMA) model
        if val_loss < best_val_loss and checkpoint_path:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": ema_state if ema_state is not None else model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_r2": val_r2.tolist(),
            }, checkpoint_path)
            wait = 0
        else:
            wait += 1

        # Restore real weights for next training epoch
        if ema_state is not None:
            model.load_state_dict(real_state)

        # Early stopping
        if patience > 0 and wait >= patience:
            print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
            break

    return history


def train_deep_ensemble(
    model_factory: callable,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    n_models: int = 5,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = None,
    output_dir: Path = None,
    parallel: bool = False,
    base_seed: int = 42,
    grad_clip: float = 1.0,
    patience: int = 20,
    use_ema: bool = True,
    ema_decay: float = 0.999,
    target_weights: Optional[torch.Tensor] = None,
    target_names: Optional[List[str]] = None,
) -> DeepEnsemble:
    """
    Train a deep ensemble with all modern best-practices.

    Each member gets:
      - AdamW optimizer with weight decay
      - Cosine-annealing LR scheduler
      - Gradient clipping
      - Early stopping
      - EMA weight averaging
      - Per-target loss weighting (if target_weights provided)
      - Per-target R² monitoring every epoch
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    loss_fn = nn.SmoothL1Loss()  # Huber loss (robust to residual outliers)
    models = []
    all_histories = []

    for i in range(n_models):
        model = model_factory()

        checkpoint_path = None
        if output_dir:
            checkpoint_path = output_dir / f"ensemble_member_{i:02d}.pt"

        # Resume: skip already-trained members
        if checkpoint_path and checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            model = model.to(device)
            r2_info = ""
            if "val_r2" in state:
                r2_vals = state["val_r2"]
                if target_names and len(r2_vals) == len(target_names):
                    r2_info = " | R²: " + ", ".join(f"{n}={v:.3f}" for n, v in zip(target_names, r2_vals))
                else:
                    r2_info = f" | R²: {r2_vals}"
            print(f"\nEnsemble member {i+1}/{n_models} - LOADED from checkpoint "
                  f"(val_loss: {state.get('val_loss', '?'):.4f} @ epoch {state.get('epoch', '?')}{r2_info})")
            models.append(model)
            continue

        print(f"\nTraining ensemble member {i+1}/{n_models}")
        import sys, traceback

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

        try:
            history = train_ensemble_member(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=epochs,
                device=device,
                seed=base_seed + i * 1000,
                checkpoint_path=checkpoint_path,
                scheduler=scheduler,
                grad_clip=grad_clip,
                patience=patience,
                use_ema=use_ema,
                ema_decay=ema_decay,
                target_weights=target_weights,
                target_names=target_names,
            )
            all_histories.append(history)
        except Exception as e:
            print(f"\n*** ENSEMBLE MEMBER {i+1} CRASHED: {e}", file=sys.stderr, flush=True)
            traceback.print_exc()
            raise

        # Load best checkpoint
        if checkpoint_path and checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            r2_info = ""
            if "val_r2" in state:
                r2_vals = state["val_r2"]
                if target_names and len(r2_vals) == len(target_names):
                    r2_info = " | R²: " + ", ".join(f"{n}={v:.3f}" for n, v in zip(target_names, r2_vals))
            print(f"  Best val_loss: {state['val_loss']:.4f} @ epoch {state['epoch']}{r2_info}")

        models.append(model)

    # Save training histories
    if output_dir and all_histories:
        import json
        with open(output_dir / "training_histories.json", "w") as f:
            json.dump(all_histories, f, indent=2)

    return DeepEnsemble(models, device=device)
