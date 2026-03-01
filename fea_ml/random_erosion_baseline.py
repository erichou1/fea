#!/usr/bin/env python3
"""
Random Erosion Baseline Experiment.

Runs the SASTO optimization with random voxel ordering (no sensitivity gradient)
to quantify the value of the sensitivity-guided approach. Uses the same:
  - 6-simple-point topology preservation
  - Conservative constraint gate (mu + k*sigma)
  - Adaptive batch halving
  - Part-aware thickness constraints
  
But replaces the neural network sensitivity ranking with random permutation.

Usage:
    cd fea_ml
    python random_erosion_baseline.py [--seeds 5] [--sample 00472]
"""

import sys, os, json, time, gc, argparse
import numpy as np
import torch
from pathlib import Path
from scipy.ndimage import (
    distance_transform_edt, label, generate_binary_structure,
    binary_dilation,
)
import yaml

sys.stdout.reconfigure(line_buffering=True)

# ── structures ──
_STRUCT26 = generate_binary_structure(3, 3)
_STRUCT6 = generate_binary_structure(3, 1)

# ── constants (same as run_batch_all.py) ──
PART_EXTERIOR_WALL = 1
PART_INTERIOR_WALL = 2
PART_ROOF = 3
PART_FLOOR = 4
NUM_PARTS = 6

MAX_VON_MISES = 5.0e6
MAX_DISPLACEMENT = 1.0
MAX_COMPLIANCE_RATIO = 1.15
UNCERTAINTY_K = 1.0
VM_IDX, DISP_IDX, COMP_IDX = 0, 1, 2

MIN_THICK_DEFAULT = 2
MIN_THICK_INTERIOR = 1
BATCH_SIZE = 200
MIN_BATCH = 10
MAX_LAYERS = 40
MAX_CONSECUTIVE_FAILS = 5
SENSITIVITY_FREQ = 3

VOLUME_WEIGHT = 1.0
SURFACE_WEIGHT = 0.01
CONSTRAINT_PENALTY = 100.0


def run_erosion(sample_id, data_dir, models, norm_dict, config, device,
                use_sensitivity=True, seed=42, verbose=True):
    """
    Run SASTO Phase 1 erosion on a single sample.
    
    Args:
        use_sensitivity: If True, use neural sensitivity ranking (SASTO).
                        If False, use random permutation (baseline).
        seed: Random seed for reproducibility (only used when use_sensitivity=False).
    
    Returns dict with results.
    """
    rng = np.random.RandomState(seed)
    
    sample_dir = data_dir / sample_id
    targets_json = json.load(open(sample_dir / "targets.json"))
    fixed_occ = np.load(sample_dir / "occ.npz")["data"].astype(np.uint8)
    fixed_part = np.load(sample_dir / "part.npz")["data"].astype(np.uint8)
    
    D, H, W = fixed_occ.shape
    original_volume = int(fixed_occ.sum())
    baseline_compliance = targets_json["compliance"]
    comp_limit = MAX_COMPLIANCE_RATIO * baseline_compliance
    
    # Normalize features
    raw_features = np.array([
        25e9/1e11, 0.20, 2400.0/1000, 30e6/1e7,
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ], dtype=np.float32)
    feat_mean = np.array(norm_dict["feature_mean"], dtype=np.float32)
    feat_std = np.array(norm_dict["feature_std"], dtype=np.float32)
    features = (raw_features - feat_mean) / (feat_std + 1e-8)
    
    target_mean = np.array(norm_dict["target_mean"], dtype=np.float32)
    target_std_arr = np.array(norm_dict["target_std"], dtype=np.float32)
    log_targets = norm_dict.get("log_transform_targets", [])
    
    def build_voxel_input(occ_arr, part_arr):
        channels = [occ_arr[None].astype(np.float32)]
        for p in range(NUM_PARTS):
            channels.append((part_arr == p).astype(np.float32)[None])
        return np.concatenate(channels, axis=0)
    
    def denorm(pred_mean, pred_std):
        pm = pred_mean * target_std_arr + target_mean
        ps = pred_std * target_std_arr
        if log_targets:
            pm_raw = np.expm1(pm)
            ps_raw = ps * np.exp(pm)
            return pm_raw, ps_raw
        return pm, ps
    
    def ensemble_predict(voxel_np, feat_np):
        preds = []
        vt = torch.from_numpy(voxel_np[None]).float().to(device)
        ft = torch.from_numpy(feat_np[None]).float().to(device)
        for model in models:
            with torch.no_grad():
                p = model(vt, ft).cpu().numpy()
            preds.append(p)
        del vt, ft
        if device.type == "cuda":
            torch.cuda.empty_cache()
        stacked = np.stack(preds, axis=0)
        return stacked.mean(axis=0).squeeze(0), stacked.std(axis=0).squeeze(0)
    
    def is_simple_point(vol, z, y, x):
        if vol[z, y, x] == 0:
            return False
        zs, ze = max(0, z-1), min(vol.shape[0], z+2)
        ys, ye = max(0, y-1), min(vol.shape[1], y+2)
        xs, xe = max(0, x-1), min(vol.shape[2], x+2)
        nb = vol[zs:ze, ys:ye, xs:xe].copy()
        cz, cy, cx = z - zs, y - ys, x - xs
        nb[cz, cy, cx] = 0
        fg_labels, fg_count = label(nb, structure=_STRUCT6)
        if fg_count != 1:
            return False
        bg = (nb == 0).astype(np.uint8)
        bg_labels, _ = label(bg, structure=_STRUCT26)
        adj_labels = set()
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    nz, ny, nx = cz + dz, cy + dy, cx + dx
                    if 0 <= nz < nb.shape[0] and 0 <= ny < nb.shape[1] and 0 <= nx < nb.shape[2]:
                        if bg[nz, ny, nx]:
                            adj_labels.add(bg_labels[nz, ny, nx])
        adj_labels.add(bg_labels[cz, cy, cx])
        return len(adj_labels) == 1
    
    def passes_thickness_check(occ, z, y, x, dt):
        min_thick = MIN_THICK_INTERIOR if fixed_part[z, y, x] == PART_INTERIOR_WALL else MIN_THICK_DEFAULT
        for dz, dy, dx in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            nz, ny, nx = z + dz, y + dy, x + dx
            if not (0 <= nz < D and 0 <= ny < H and 0 <= nx < W):
                continue
            if occ[nz, ny, nx] > 0:
                continue
            bz, by, bx = z - dz, y - dy, x - dx
            if not (0 <= bz < D and 0 <= by < H and 0 <= bx < W):
                return False
            if occ[bz, by, bx] == 0:
                return False
            if dt[bz, by, bx] < min_thick:
                return False
        return True
    
    def classify_air(occ_vol):
        air = (occ_vol == 0)
        air_labels, n_air = label(air, structure=_STRUCT6)
        boundary_labels = set()
        for face in [air_labels[0,:,:], air_labels[-1,:,:],
                     air_labels[:,0,:], air_labels[:,-1,:],
                     air_labels[:,:,0], air_labels[:,:,-1]]:
            boundary_labels.update(face[face > 0].tolist())
        interior_air = np.zeros_like(air)
        for lbl in range(1, n_air + 1):
            if lbl not in boundary_labels:
                interior_air[air_labels == lbl] = True
        return interior_air
    
    def find_interior_surface(occ_vol, interior_air):
        dilated = binary_dilation(interior_air, structure=_STRUCT6, iterations=1)
        return (occ_vol > 0) & dilated
    
    def compute_sensitivity(current_occ):
        masked_part = fixed_part.copy()
        masked_part[current_occ == 0] = 0
        vi = build_voxel_input(current_occ, masked_part)
        ft = torch.from_numpy(features[None]).float().to(device)
        grad_total = np.zeros((D, H, W), dtype=np.float64)
        for model in models:
            vt = torch.from_numpy(vi[None]).float().to(device)
            vt.requires_grad_(True)
            with torch.no_grad():
                pass  # warmup not needed
            pred = model(vt, ft)
            loss = pred[0, COMP_IDX] + 0.3 * pred[0, VM_IDX]
            loss.backward()
            grad_total += vt.grad[0, 0].detach().cpu().numpy()
            del vt, pred, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del ft
        grad_total /= len(models)
        return grad_total.astype(np.float32)
    
    def evaluate_occ(current_occ):
        volume = int(current_occ.sum())
        masked_part = fixed_part.copy()
        masked_part[current_occ == 0] = 0
        vi = build_voxel_input(current_occ, masked_part)
        pmean, pstd = ensemble_predict(vi, features)
        pmean_d, pstd_d = denorm(pmean, pstd)
        vm_c = pmean_d[VM_IDX] + UNCERTAINTY_K * pstd_d[VM_IDX]
        disp_c = pmean_d[DISP_IDX] + UNCERTAINTY_K * pstd_d[DISP_IDX]
        comp_c = pmean_d[COMP_IDX] + UNCERTAINTY_K * pstd_d[COMP_IDX]
        vm_viol = max(0, vm_c - MAX_VON_MISES) / max(MAX_VON_MISES, 1e-12)
        disp_viol = max(0, disp_c - MAX_DISPLACEMENT)
        comp_viol = max(0, comp_c - comp_limit) / comp_limit if comp_limit else 0.0
        constraints_ok = (vm_viol == 0) and (disp_viol == 0) and (comp_viol == 0)
        del masked_part, vi
        return {"volume": volume, "constraints_ok": constraints_ok,
                "vm_conservative": vm_c, "comp_conservative": comp_c,
                "disp_conservative": disp_c}
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 1: EROSION (sensitivity-guided or random)
    # ═══════════════════════════════════════════════════════════
    current_occ = fixed_occ.copy()
    total_removed = 0
    batch_num = 0
    accepted_batches = 0
    rejected_batches = 0
    consecutive_fails = 0
    sensitivity = None
    method_name = "SASTO-PA" if use_sensitivity else "Random"
    
    t_start = time.time()
    
    for layer in range(MAX_LAYERS):
        dt = distance_transform_edt(current_occ > 0)
        int_air = classify_air(current_occ)
        if int(int_air.sum()) == 0:
            break
        
        cand_mask = find_interior_surface(current_occ, int_air)
        cand_coords_all = np.argwhere(cand_mask)
        if len(cand_coords_all) == 0:
            break
        
        approved = []
        for z, y, x in cand_coords_all:
            if passes_thickness_check(current_occ, z, y, x, dt):
                approved.append((z, y, x))
        if not approved:
            break
        
        if use_sensitivity:
            # Sensitivity-guided ranking
            if sensitivity is None or layer % SENSITIVITY_FREQ == 0:
                sensitivity = compute_sensitivity(current_occ)
            scores = np.array([sensitivity[z, y, x] for z, y, x in approved])
            sorted_idx = np.argsort(-scores)
            approved = [approved[i] for i in sorted_idx]
        else:
            # RANDOM ranking
            rng.shuffle(approved)
        
        ptr = 0
        current_batch_size = BATCH_SIZE
        while ptr < len(approved) and consecutive_fails < MAX_CONSECUTIVE_FAILS:
            batch_num += 1
            simple_pts = []
            while len(simple_pts) < current_batch_size and ptr < len(approved):
                z, y, x = approved[ptr]
                ptr += 1
                if current_occ[z, y, x] == 0:
                    continue
                if is_simple_point(current_occ, z, y, x):
                    simple_pts.append((z, y, x))
                    current_occ[z, y, x] = 0
            if not simple_pts:
                continue
            
            result = evaluate_occ(current_occ)
            
            if result["constraints_ok"]:
                total_removed += len(simple_pts)
                consecutive_fails = 0
                current_batch_size = BATCH_SIZE
                accepted_batches += 1
            else:
                for z, y, x in simple_pts:
                    current_occ[z, y, x] = 1
                consecutive_fails += 1
                current_batch_size = max(MIN_BATCH, current_batch_size // 2)
                rejected_batches += 1
        
        if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
            break
    
    elapsed = time.time() - t_start
    final_volume = int(current_occ.sum())
    vol_reduction = 1.0 - final_volume / original_volume
    rejection_rate = rejected_batches / max(1, batch_num)
    
    if verbose:
        print(f"  [{method_name}] seed={seed}: "
              f"reduction={100*vol_reduction:.1f}%, "
              f"batches={batch_num} (accept={accepted_batches}, reject={rejected_batches}, "
              f"rejection_rate={100*rejection_rate:.1f}%), "
              f"time={elapsed:.1f}s")
    
    return {
        "method": method_name,
        "seed": seed,
        "sample_id": sample_id,
        "original_volume": original_volume,
        "final_volume": final_volume,
        "total_removed": total_removed,
        "volume_reduction_pct": 100.0 * vol_reduction,
        "n_batches": batch_num,
        "accepted_batches": accepted_batches,
        "rejected_batches": rejected_batches,
        "rejection_rate": rejection_rate,
        "time_seconds": elapsed,
        "constraints_satisfied": vol_reduction > 0.001,
    }


def main():
    parser = argparse.ArgumentParser(description="Random erosion baseline experiment")
    parser.add_argument("--sample", default="00472", help="Sample ID to test")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--extra-samples", nargs="*", default=[],
                        help="Additional sample IDs to test")
    args = parser.parse_args()
    
    base = Path(__file__).parent
    data_dir = base / "data" / "runs_real_128"
    config_path = base / "runs" / "v3" / "config.yaml"
    ensemble_dir = base / "runs" / "v3" / "ensemble"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load ensemble
    config = yaml.safe_load(open(config_path))
    norm_dict = json.load(open(config_path.parent / "normalization.json"))
    
    from fea_ml.models.cnn3d import Surrogate3DResNet
    model_cfg = config.get("model", {})
    members = sorted(ensemble_dir.glob("ensemble_member_*.pt"))
    models = []
    for mp in members:
        ckpt = torch.load(mp, map_location="cpu", weights_only=False)
        m = Surrogate3DResNet(
            in_channels=7, feature_dim=10,
            target_dim=len(config["targets"]),
            base_channels=model_cfg.get("base_channels", 64),
            dropout=model_cfg.get("dropout", 0.15),
            drop_path=model_cfg.get("drop_path", 0.1),
        )
        state = ckpt.get("model_state_dict", ckpt.get("ema_state_dict", ckpt))
        m.load_state_dict(state)
        m.eval()
        m.to(device)
        models.append(m)
        del ckpt, state
    print(f"Loaded {len(models)} ensemble members -> {device}")
    
    all_samples = [args.sample] + args.extra_samples
    all_results = []
    
    for sample_id in all_samples:
        print(f"\n{'='*60}")
        print(f"Sample: {sample_id}")
        print(f"{'='*60}")
        
        # Run SASTO (sensitivity-guided)
        print("\n--- SASTO-PA (sensitivity-guided) ---")
        sasto_result = run_erosion(
            sample_id, data_dir, models, norm_dict, config, device,
            use_sensitivity=True, seed=0, verbose=True
        )
        all_results.append(sasto_result)
        
        # Run Random baseline (multiple seeds)
        print(f"\n--- Random Erosion Baseline ({args.seeds} seeds) ---")
        random_results = []
        for seed in range(args.seeds):
            result = run_erosion(
                sample_id, data_dir, models, norm_dict, config, device,
                use_sensitivity=False, seed=seed, verbose=True
            )
            random_results.append(result)
            all_results.append(result)
        
        # Summary
        sasto_red = sasto_result["volume_reduction_pct"]
        random_reds = [r["volume_reduction_pct"] for r in random_results]
        random_rejects = [100 * r["rejection_rate"] for r in random_results]
        sasto_reject = 100 * sasto_result["rejection_rate"]
        
        print(f"\n--- Summary for {sample_id} ---")
        print(f"  SASTO-PA:  {sasto_red:.1f}% reduction, "
              f"{sasto_result['n_batches']} batches, "
              f"{sasto_reject:.1f}% rejection rate, "
              f"{sasto_result['time_seconds']:.1f}s")
        print(f"  Random:    {np.mean(random_reds):.1f}% ± {np.std(random_reds):.1f}% reduction "
              f"(range [{min(random_reds):.1f}%, {max(random_reds):.1f}%]), "
              f"{np.mean(random_rejects):.1f}% rejection rate, "
              f"{np.mean([r['time_seconds'] for r in random_results]):.1f}s avg")
        print(f"  Improvement: SASTO achieves "
              f"{sasto_red - np.mean(random_reds):.1f} pp more reduction")
    
    # Save results
    out_path = base / "runs" / "v3" / "random_erosion_baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
