"""
Training monitor for parallel ensemble training.

Reads per-member history JSON files and generates comprehensive
training progress plots.  Can be run periodically while training is
ongoing (e.g. in a ``watch`` loop).

Usage:
    # One-shot plot generation
    python -m fea_ml_hires.scripts.monitor_training runs/hires_512_v1

    # Continuous monitoring (re-plots every 60s)
    python -m fea_ml_hires.scripts.monitor_training runs/hires_512_v1 --watch 60

    # TensorBoard (always available alongside this)
    tensorboard --logdir runs/hires_512_v1/tensorboard
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_histories(output_dir: Path) -> Dict[int, Dict]:
    """Load all member training histories."""
    log_dir = output_dir / "logs"
    histories = {}
    for f in sorted(log_dir.glob("member_*_history.json")):
        try:
            member_id = int(f.stem.split("_")[1])
            with open(f) as fh:
                histories[member_id] = json.load(fh)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"  Warning: couldn't parse {f.name}: {e}")
    return histories


def load_config(output_dir: Path) -> Dict:
    """Load saved config."""
    config_path = output_dir / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def load_checkpoints(output_dir: Path) -> Dict[int, Dict]:
    """Load best checkpoint metadata for each member."""
    ensemble_dir = output_dir / "ensemble"
    results = {}
    for f in sorted(ensemble_dir.glob("ensemble_member_*.pt")):
        try:
            import torch
            member_id = int(f.stem.split("_")[-1])
            state = torch.load(f, map_location="cpu", weights_only=False)
            results[member_id] = {
                "epoch": state.get("epoch", -1),
                "val_loss": state.get("val_loss", -1),
                "val_r2": state.get("val_r2", []),
                "mae": state.get("mae", []),
                "rmse": state.get("rmse", []),
            }
        except Exception:
            pass
    return results


def generate_plots(output_dir: Path) -> None:
    """Generate comprehensive training monitoring plots."""
    if not HAS_MPL:
        print("matplotlib not installed — cannot generate plots")
        return

    histories = load_histories(output_dir)
    if not histories:
        print("No training histories found yet.")
        return

    config = load_config(output_dir)
    target_names = config.get("targets", ["max_von_mises", "max_displacement",
                                           "min_safety_factor", "compliance"])
    n_members = len(histories)

    fig = plt.figure(figsize=(24, 18))
    fig.suptitle(f"Ensemble Training Monitor — {n_members} members\n{output_dir}",
                 fontsize=14, fontweight="bold")
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    colors = plt.cm.tab10(np.linspace(0, 1, max(n_members, 1)))

    # ---- 1. Training Loss (all members) ----
    ax1 = fig.add_subplot(gs[0, 0])
    for mid, h in sorted(histories.items()):
        ax1.plot(h["train_loss"], color=colors[mid % len(colors)],
                 alpha=0.7, label=f"M{mid}")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_yscale("log")
    ax1.legend(fontsize=6, ncol=2)
    ax1.grid(True, alpha=0.3)

    # ---- 2. Validation Loss (all members) ----
    ax2 = fig.add_subplot(gs[0, 1])
    for mid, h in sorted(histories.items()):
        ax2.plot(h["val_loss"], color=colors[mid % len(colors)],
                 alpha=0.7, label=f"M{mid}")
    ax2.set_title("Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_yscale("log")
    ax2.legend(fontsize=6, ncol=2)
    ax2.grid(True, alpha=0.3)

    # ---- 3–6. Per-target R² (one subplot per target) ----
    for t_idx, t_name in enumerate(target_names):
        row = (t_idx // 2) + 1
        col = (t_idx % 2)
        ax = fig.add_subplot(gs[row, col])
        for mid, h in sorted(histories.items()):
            r2_series = [ep_r2[t_idx] if t_idx < len(ep_r2) else 0
                         for ep_r2 in h.get("val_r2", [])]
            ax.plot(r2_series, color=colors[mid % len(colors)],
                    alpha=0.7, label=f"M{mid}")
        ax.set_title(f"R² — {t_name}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("R²")
        ax.axhline(y=0.9, color="green", linestyle="--", alpha=0.3, label="0.9")
        ax.axhline(y=0.95, color="blue", linestyle="--", alpha=0.3, label="0.95")
        ax.set_ylim(-0.5, 1.05)
        ax.legend(fontsize=5, ncol=2)
        ax.grid(True, alpha=0.3)

    # ---- 7. Learning Rate ----
    ax7 = fig.add_subplot(gs[0, 2])
    for mid, h in sorted(histories.items()):
        if "lr" in h and h["lr"]:
            ax7.plot(h["lr"], color=colors[mid % len(colors)],
                     alpha=0.7, label=f"M{mid}")
    ax7.set_title("Learning Rate")
    ax7.set_xlabel("Epoch")
    ax7.set_ylabel("LR")
    ax7.set_yscale("log")
    ax7.legend(fontsize=6, ncol=2)
    ax7.grid(True, alpha=0.3)

    # ---- 8. Gradient Norms ----
    ax8 = fig.add_subplot(gs[0, 3])
    for mid, h in sorted(histories.items()):
        if "grad_norm" in h and h["grad_norm"]:
            ax8.plot(h["grad_norm"], color=colors[mid % len(colors)],
                     alpha=0.7, label=f"M{mid}")
    ax8.set_title("Gradient Norm")
    ax8.set_xlabel("Epoch")
    ax8.set_ylabel("‖∇‖")
    ax8.legend(fontsize=6, ncol=2)
    ax8.grid(True, alpha=0.3)

    # ---- 9. GPU Memory ----
    ax9 = fig.add_subplot(gs[1, 2])
    for mid, h in sorted(histories.items()):
        if "gpu_mem_gb" in h and h["gpu_mem_gb"]:
            ax9.plot(h["gpu_mem_gb"], color=colors[mid % len(colors)],
                     alpha=0.7, label=f"M{mid}")
    ax9.set_title("GPU Memory (GB)")
    ax9.set_xlabel("Epoch")
    ax9.set_ylabel("GB")
    ax9.legend(fontsize=6, ncol=2)
    ax9.grid(True, alpha=0.3)

    # ---- 10. Epoch Time ----
    ax10 = fig.add_subplot(gs[1, 3])
    for mid, h in sorted(histories.items()):
        if "epoch_time" in h and h["epoch_time"]:
            ax10.plot(h["epoch_time"], color=colors[mid % len(colors)],
                      alpha=0.7, label=f"M{mid}")
    ax10.set_title("Epoch Time (s)")
    ax10.set_xlabel("Epoch")
    ax10.set_ylabel("Seconds")
    ax10.legend(fontsize=6, ncol=2)
    ax10.grid(True, alpha=0.3)

    # ---- 11. Best R² summary bar chart ----
    ax11 = fig.add_subplot(gs[2, 2:])
    checkpoints = load_checkpoints(output_dir)
    if checkpoints:
        member_ids = sorted(checkpoints.keys())
        x = np.arange(len(target_names))
        width = 0.8 / len(member_ids)
        for i, mid in enumerate(member_ids):
            r2 = checkpoints[mid].get("val_r2", [0] * len(target_names))
            r2 = r2[:len(target_names)]
            while len(r2) < len(target_names):
                r2.append(0)
            ax11.bar(x + i * width, r2, width, label=f"M{mid}",
                     color=colors[mid % len(colors)], alpha=0.8)
        ax11.set_title("Best R² per Member (at checkpoint)")
        ax11.set_xticks(x + width * len(member_ids) / 2)
        ax11.set_xticklabels(target_names, fontsize=8)
        ax11.set_ylabel("R²")
        ax11.axhline(y=0.9, color="green", linestyle="--", alpha=0.3)
        ax11.legend(fontsize=6, ncol=4)
        ax11.grid(True, alpha=0.3, axis="y")

    # Save
    plot_path = output_dir / "training_monitor.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {plot_path}")


def print_status(output_dir: Path) -> None:
    """Print a text summary of training progress."""
    histories = load_histories(output_dir)
    checkpoints = load_checkpoints(output_dir)
    config = load_config(output_dir)
    target_names = config.get("targets", ["vm", "disp", "sf", "comp"])
    n_expected = config.get("model", {}).get("n_models", 8)

    print(f"\n{'='*70}")
    print(f"  TRAINING STATUS — {output_dir}")
    print(f"{'='*70}")

    # Active training
    for mid in sorted(histories.keys()):
        h = histories[mid]
        n_epochs = len(h["train_loss"])
        if n_epochs == 0:
            continue
        last_train = h["train_loss"][-1]
        last_val = h["val_loss"][-1]
        last_r2 = h["val_r2"][-1] if h["val_r2"] else []
        r2_str = ", ".join(f"{target_names[i]}={last_r2[i]:.4f}"
                          for i in range(min(len(target_names), len(last_r2))))
        status = "DONE" if mid in checkpoints else "TRAINING"
        print(f"  M{mid:02d} [{status:8s}] ep={n_epochs:3d} | "
              f"train={last_train:.4f} val={last_val:.4f} | R²=[{r2_str}]")

    # Missing members
    trained = set(histories.keys())
    for m in range(n_expected):
        if m not in trained:
            if m in checkpoints:
                print(f"  M{m:02d} [LOADED   ] from previous run")
            else:
                print(f"  M{m:02d} [PENDING  ]")

    # Best results
    if checkpoints:
        print(f"\n  Best checkpoint results:")
        for mid in sorted(checkpoints.keys()):
            cp = checkpoints[mid]
            r2 = cp.get("val_r2", [])
            r2_str = ", ".join(f"{target_names[i]}={r2[i]:.4f}"
                              for i in range(min(len(target_names), len(r2))))
            print(f"    M{mid:02d}: val_loss={cp['val_loss']:.4f}, "
                  f"epoch={cp['epoch']}, R²=[{r2_str}]")

        # Ensemble mean R²
        all_r2 = np.array([cp["val_r2"] for cp in checkpoints.values()
                           if cp.get("val_r2")])
        if len(all_r2) > 0:
            mean_r2 = all_r2.mean(axis=0)
            std_r2 = all_r2.std(axis=0)
            print(f"\n  Ensemble mean R²:")
            for i, tn in enumerate(target_names):
                if i < len(mean_r2):
                    print(f"    {tn}: {mean_r2[i]:.4f} ± {std_r2[i]:.4f}")

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Monitor ensemble training progress")
    parser.add_argument("output_dir", type=str, help="Training output directory")
    parser.add_argument("--watch", type=int, default=None,
                        help="Re-run every N seconds (continuous monitoring)")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip plot generation (text-only)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        sys.exit(1)

    if args.watch:
        print(f"Monitoring every {args.watch}s (Ctrl+C to stop)")
        while True:
            try:
                print_status(output_dir)
                if not args.no_plot:
                    generate_plots(output_dir)
                time.sleep(args.watch)
            except KeyboardInterrupt:
                print("\nMonitoring stopped.")
                break
    else:
        print_status(output_dir)
        if not args.no_plot:
            generate_plots(output_dir)


if __name__ == "__main__":
    main()
