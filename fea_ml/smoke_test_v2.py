"""Comprehensive smoke test for v2 training pipeline."""
import torch
import numpy as np
import json
import time
from pathlib import Path
from fea_ml.data.voxel_dataset import (
    VoxelFEADataset,
    compute_voxel_normalization_stats,
    create_data_splits,
)
from fea_ml.models.cnn3d import create_surrogate_model
from fea_ml.utils.config import load_config
from torch.utils.data import DataLoader


def main():
    config = load_config("configs/voxel_config.yaml")

    # Use local 64-res data
    runs_dir = Path("data/runs_real")
    train_dirs, val_dirs, test_dirs = create_data_splits(
        runs_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42,
        split_by_family=True,
    )
    print(f"Splits: train={len(train_dirs)}, val={len(val_dirs)}, test={len(test_dirs)}")

    # Small subset for speed
    train_sub = train_dirs[:20]
    val_sub = val_dirs[:5]

    target_names = tuple(config["targets"])
    materials = tuple(config["materials"])
    load_cases = tuple(config["load_cases"])

    # Create dataset
    ds = VoxelFEADataset(train_sub, target_names, materials, load_cases, resolution=64, augment=True)
    stats = compute_voxel_normalization_stats(ds, log_transform_targets=None, winsorize_percentile=2.0)
    ds.stats = stats

    print(f"Normalization: log_targets={stats.log_transform_targets}")
    print(f"  target_mean={stats.target_mean}")
    print(f"  target_std ={stats.target_std}")

    # Verify normalization produces reasonable values
    sample = ds[0]
    norm_t = sample["targets"].numpy()
    print(f"  sample targets (normalized): {norm_t}")
    assert all(abs(v) < 10 for v in norm_t), f"Normalized targets out of range! {norm_t}"

    val_ds = VoxelFEADataset(val_sub, target_names, materials, load_cases, resolution=64, stats=stats, augment=False)

    train_loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    # Create model (64-res version)
    model = create_surrogate_model(
        in_channels=ds.get_voxel_channels(),
        feature_dim=ds.get_feature_dim(),
        target_dim=len(target_names),
        resolution=64,
        dropout=0.15,
        drop_path=0.1,
        backbone="cnn3d",
        base_channels=32,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}, device: {device}")

    # Quick training loop (2 epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    loss_fn = torch.nn.SmoothL1Loss()

    for epoch in range(2):
        model.train()
        total = 0
        n = 0
        for batch in train_loader:
            v = batch["voxel"].to(device)
            f = batch["features"].to(device)
            t = batch["targets"].to(device)
            optimizer.zero_grad()
            pred = model(v, f)
            loss = loss_fn(pred, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            n += 1

        # Validate
        model.eval()
        val_loss = 0
        vn = 0
        all_p = []
        all_t = []
        with torch.no_grad():
            for batch in val_loader:
                v = batch["voxel"].to(device)
                f = batch["features"].to(device)
                t = batch["targets"].to(device)
                p = model(v, f)
                val_loss += loss_fn(p, t).item()
                vn += 1
                all_p.append(p.cpu().numpy())
                all_t.append(t.cpu().numpy())

        preds = np.concatenate(all_p)
        targs = np.concatenate(all_t)
        r2s = []
        for i in range(preds.shape[1]):
            ss_res = ((targs[:, i] - preds[:, i]) ** 2).sum()
            ss_tot = ((targs[:, i] - targs[:, i].mean()) ** 2).sum()
            r2s.append(1 - ss_res / (ss_tot + 1e-8))

        print(
            f"Epoch {epoch+1}: train_loss={total/n:.4f}, val_loss={val_loss/vn:.4f}, "
            f"R2={[f'{r:.3f}' for r in r2s]}"
        )

    # Test denormalization round-trip
    print("\n=== Denormalization test ===")
    raw_target = np.array([2.18e6, 31000.0, 55000.0, 0.14], dtype=np.float64)
    log_vals = np.log1p(np.abs(raw_target))
    z_vals = (log_vals - stats.target_mean.astype(np.float64)) / stats.target_std.astype(np.float64)
    # Reverse
    log_back = z_vals * stats.target_std.astype(np.float64) + stats.target_mean.astype(np.float64)
    raw_back = np.expm1(log_back)
    print(f"Original:       {raw_target}")
    print(f"Round-trip:     {raw_back}")
    rel_err = np.abs(raw_target - raw_back) / (np.abs(raw_target) + 1e-12)
    print(f"Relative error: {rel_err}")
    assert all(e < 1e-4 for e in rel_err), f"Denormalization round-trip error too large! {rel_err}"

    # Test NormalizationStats serialization round-trip
    print("\n=== NormalizationStats JSON round-trip ===")
    d = stats.to_dict()
    from fea_ml.data.voxel_dataset import VoxelNormalizationStats
    stats2 = VoxelNormalizationStats.from_dict(d)
    assert np.allclose(stats.target_mean, stats2.target_mean)
    assert np.allclose(stats.target_std, stats2.target_std)
    assert stats.log_transform_targets == stats2.log_transform_targets
    print("OK: JSON round-trip matches")

    # Test 128-res model creation
    print("\n=== 128-res ResNet model ===")
    model128 = create_surrogate_model(
        in_channels=7, feature_dim=13, target_dim=4,
        resolution=128, dropout=0.15, drop_path=0.1,
        backbone="resnet3d", base_channels=64,
    )
    n128 = sum(p.numel() for p in model128.parameters())
    print(f"ResNet 128³: {n128:,} params")
    # Forward pass with dummy data
    dummy_v = torch.randn(1, 7, 128, 128, 128)
    dummy_f = torch.randn(1, 13)
    with torch.no_grad():
        out = model128(dummy_v, dummy_f)
    print(f"Forward pass: input=128³ → output={out.shape}")
    assert out.shape == (1, 4), f"Wrong output shape: {out.shape}"

    print("\n" + "=" * 50)
    print("ALL SMOKE TESTS PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    main()
