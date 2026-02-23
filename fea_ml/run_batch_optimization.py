#!/usr/bin/env python3
"""
Batch SASTO Optimization — Run Part-Aware optimization on multiple house models.

Runs the SASTO sensitivity-guided topology optimization (formerly V11)
on a list of sample IDs and collects aggregate results.

This is a refactored, batch-compatible version of run_opt_v11.py.

Usage:
    cd fea_ml
    python run_batch_optimization.py

Output: runs/v3/batch_results/
"""
import sys, os, json, time, gc, traceback, faulthandler
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
import torch
from pathlib import Path
from scipy.ndimage import (
    distance_transform_edt, label, generate_binary_structure,
    binary_dilation,
)
from scipy.signal import fftconvolve
import yaml

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
faulthandler.enable()

# ── topology structures ──────────────────────────────────────────
_STRUCT26 = generate_binary_structure(3, 3)
_STRUCT6  = generate_binary_structure(3, 1)

# ── constants ────────────────────────────────────────────────────
PART_EXTERIOR_WALL = 1
PART_INTERIOR_WALL = 2
PART_ROOF  = 3
PART_FLOOR = 4
NUM_PARTS  = 6

# ── optimization constraints ────────────────────────────────────
VOLUME_WEIGHT      = 1.0
SURFACE_WEIGHT     = 0.01
CONSTRAINT_PENALTY = 100.0
MAX_VON_MISES      = 5.0e6
MAX_DISPLACEMENT   = 1.0
MAX_COMPLIANCE_RATIO = 1.15
UNCERTAINTY_K      = 1.0
VM_IDX, DISP_IDX, COMP_IDX = 0, 1, 2

# ── optimization parameters ─────────────────────────────────────
MIN_THICK_DEFAULT  = 2
MIN_THICK_INTERIOR = 1
BATCH_SIZE         = 200
MIN_BATCH          = 10
MAX_LAYERS         = 40
MAX_CONSECUTIVE_FAILS = 5
SENSITIVITY_FREQ   = 3
ENDGAME_BATCH      = 5
ENDGAME_MAX_EVALS  = 200
SWAP_MAX_ATTEMPTS  = 50


# ── 20 diverse test samples (spread across volume range) ─────────
TEST_SAMPLES = [
    '15935', '10936', '12076', '09857', '08739',
    '08288', '01845', '10662', '00037', '00739',
    '05153', '14283', '13430', '10792', '13005',
    '12641', '00777', '08236', '12735', '04062',
]


def load_ensemble(config_path, ensemble_dir, device):
    """Load config, normalization, and ensemble models."""
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
        gc.collect()
    print(f"Loaded {len(models)} ensemble members -> {device}")
    return config, norm_dict, models


def optimize_sample(sample_id, data_dir, output_dir, models, norm_dict,
                    config, device, verbose=True):
    """
    Run SASTO Part-Aware optimization on a single sample.

    Returns dict with all result metrics, or None on failure.
    """
    sample_dir = data_dir / sample_id
    if not sample_dir.exists():
        print(f"  [SKIP] {sample_id}: directory not found")
        return None

    out_dir = output_dir / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # ── load sample ───────────────────────────────────────────
    targets_json = json.load(open(sample_dir / "targets.json"))
    fixed_occ = np.load(sample_dir / "occ.npz")["data"].astype(np.uint8)
    fixed_part = np.load(sample_dir / "part.npz")["data"].astype(np.uint8)
    D, H, W = fixed_occ.shape
    original_volume = int(fixed_occ.sum())

    baseline_compliance = targets_json.get("compliance", None)
    comp_limit = MAX_COMPLIANCE_RATIO * baseline_compliance if baseline_compliance else None

    if verbose:
        print(f"  Sample {sample_id}: {original_volume:,} voxels, "
              f"comp_limit={comp_limit:.4f}")

    # ── features ──────────────────────────────────────────────
    raw_features = np.array([
        25e9/1e11, 0.20, 2400.0/1000, 30e6/1e7,
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ], dtype=np.float32)
    feat_mean = np.array(norm_dict["feature_mean"], dtype=np.float32)
    feat_std  = np.array(norm_dict["feature_std"],  dtype=np.float32)
    features  = (raw_features - feat_mean) / (feat_std + 1e-8)

    target_mean    = np.array(norm_dict["target_mean"], dtype=np.float32)
    target_std_arr = np.array(norm_dict["target_std"],  dtype=np.float32)
    log_targets    = norm_dict.get("log_transform_targets", [])

    # ── helper functions ──────────────────────────────────────
    def build_voxel_input(occ_arr, part_arr):
        channels = [occ_arr[None].astype(np.float32)]
        for p in range(NUM_PARTS):
            channels.append((part_arr == p).astype(np.float32)[None])
        return np.concatenate(channels, axis=0)

    def compute_surface_area_local(occ_arr):
        padded = np.pad(occ_arr, 1, mode='constant', constant_values=0)
        return (int(np.sum(np.abs(np.diff(padded, axis=0)))) +
                int(np.sum(np.abs(np.diff(padded, axis=1)))) +
                int(np.sum(np.abs(np.diff(padded, axis=2)))))

    def denorm(pred_mean, pred_std):
        pm = pred_mean * target_std_arr + target_mean
        ps = pred_std  * target_std_arr
        if log_targets:
            pm_raw = np.expm1(pm)
            ps_raw = ps * np.exp(pm)
            return pm_raw, ps_raw
        return pm, ps

    def ensemble_predict(voxel_np, feat_np):
        vt = torch.from_numpy(voxel_np[None]).float().to(device)
        ft = torch.from_numpy(feat_np[None]).float().to(device)
        preds = []
        for model in models:
            with torch.no_grad():
                p = model(vt, ft).cpu().numpy()
            preds.append(p)
        del vt, ft
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

    def get_min_thick(z, y, x):
        if fixed_part[z, y, x] == PART_INTERIOR_WALL:
            return MIN_THICK_INTERIOR
        return MIN_THICK_DEFAULT

    def passes_thickness_check(occ, z, y, x, dt):
        min_thick = get_min_thick(z, y, x)
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
        for i, model in enumerate(models):
            vt = torch.from_numpy(vi[None]).float().to(device)
            vt.requires_grad_(True)
            try:
                pred = model(vt, ft)
                loss = pred[0, COMP_IDX] + 0.3 * pred[0, VM_IDX]
                loss.backward()
                grad_total += vt.grad[0, 0].detach().cpu().numpy()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    m_cpu = model.cpu()
                    vt_c = torch.from_numpy(vi[None]).float().requires_grad_(True)
                    ft_c = torch.from_numpy(features[None]).float()
                    pred = m_cpu(vt_c, ft_c)
                    loss = pred[0, COMP_IDX] + 0.3 * pred[0, VM_IDX]
                    loss.backward()
                    grad_total += vt_c.grad[0, 0].detach().numpy()
                    model.to(device)
                    del vt_c, ft_c, pred, loss
                else:
                    raise
            finally:
                del vt
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
        del ft
        grad_total /= len(models)
        return grad_total.astype(np.float32)

    def evaluate_occ(current_occ):
        volume = int(current_occ.sum())
        sa = compute_surface_area_local(current_occ)
        masked_part = fixed_part.copy()
        masked_part[current_occ == 0] = 0
        vi = build_voxel_input(current_occ, masked_part)
        pmean, pstd = ensemble_predict(vi, features)
        pmean_d, pstd_d = denorm(pmean, pstd)
        vm_c   = pmean_d[VM_IDX]   + UNCERTAINTY_K * pstd_d[VM_IDX]
        disp_c = pmean_d[DISP_IDX] + UNCERTAINTY_K * pstd_d[DISP_IDX]
        comp_c = pmean_d[COMP_IDX] + UNCERTAINTY_K * pstd_d[COMP_IDX]
        vm_viol   = max(0, vm_c   - MAX_VON_MISES)  / max(MAX_VON_MISES, 1e-12)
        disp_viol = max(0, disp_c - MAX_DISPLACEMENT)
        comp_viol = max(0, comp_c - comp_limit) / comp_limit if comp_limit else 0.0
        constraints_ok = (vm_viol == 0) and (disp_viol == 0) and (comp_viol == 0)
        norm_vol = volume / max(original_volume, 1)
        norm_sa  = sa     / max(original_volume, 1)
        obj = VOLUME_WEIGHT * norm_vol + SURFACE_WEIGHT * norm_sa
        if not constraints_ok:
            obj += CONSTRAINT_PENALTY * (vm_viol + disp_viol + comp_viol)
        del masked_part, vi
        gc.collect()
        return {
            "obj": obj, "volume": volume,
            "pred_mean": pmean_d, "pred_std": pstd_d,
            "constraints_ok": constraints_ok,
            "vm_conservative": vm_c, "disp_conservative": disp_c,
            "comp_conservative": comp_c,
        }

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: SENSITIVITY-GUIDED BATCH EROSION
    # ═══════════════════════════════════════════════════════════
    current_occ = fixed_occ.copy()
    total_removed = 0
    batch_num = 0
    history = []
    consecutive_fails = 0
    sensitivity = None

    for layer in range(MAX_LAYERS):
        dt = distance_transform_edt(current_occ > 0)
        int_air = classify_air(current_occ)
        if int(int_air.sum()) == 0:
            break

        cand_mask = find_interior_surface(current_occ, int_air)
        cand_coords_all = np.argwhere(cand_mask)
        if len(cand_coords_all) == 0:
            break

        # Thickness filter
        approved = []
        for z, y, x in cand_coords_all:
            if passes_thickness_check(current_occ, z, y, x, dt):
                approved.append((z, y, x))
        if not approved:
            break

        # Sensitivity sorting
        if sensitivity is None or layer % SENSITIVITY_FREQ == 0:
            sensitivity = compute_sensitivity(current_occ)
        scores = np.array([sensitivity[z, y, x] for z, y, x in approved])
        sorted_idx = np.argsort(-scores)
        approved = [approved[i] for i in sorted_idx]

        # Batch processing
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
            vol_red = 1.0 - result["volume"] / original_volume

            if result["constraints_ok"]:
                total_removed += len(simple_pts)
                consecutive_fails = 0
                current_batch_size = BATCH_SIZE
                entry = {
                    "batch": batch_num, "layer": layer + 1, "phase": "erosion",
                    "removed": len(simple_pts), "total_removed": total_removed,
                    "volume": result["volume"],
                    "vol_reduction": float(vol_red),
                    "obj": float(result["obj"]),
                    "vm":  float(result["vm_conservative"]),
                    "disp": float(result["disp_conservative"]),
                    "comp": float(result["comp_conservative"]),
                }
                history.append(entry)
            else:
                for z, y, x in simple_pts:
                    current_occ[z, y, x] = 1
                consecutive_fails += 1
                current_batch_size = max(MIN_BATCH, current_batch_size // 2)

        if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
            break

    phase1_vol = int(current_occ.sum())

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: END-GAME
    # ═══════════════════════════════════════════════════════════
    endgame_evals = 0
    endgame_removed = 0
    consecutive_fails = 0
    sensitivity = compute_sensitivity(current_occ)

    for eg_round in range(20):
        dt = distance_transform_edt(current_occ > 0)
        int_air = classify_air(current_occ)
        if int(int_air.sum()) == 0:
            break

        cand_mask = find_interior_surface(current_occ, int_air)
        approved = [(z, y, x) for z, y, x in np.argwhere(cand_mask)
                    if passes_thickness_check(current_occ, z, y, x, dt)]
        if not approved:
            break

        scores = np.array([sensitivity[z, y, x] for z, y, x in approved])
        sorted_idx = np.argsort(-scores)
        approved = [approved[i] for i in sorted_idx]
        ptr = 0

        while ptr < len(approved) and endgame_evals < ENDGAME_MAX_EVALS:
            batch_num += 1
            simple_pts = []
            while len(simple_pts) < ENDGAME_BATCH and ptr < len(approved):
                z, y, x = approved[ptr]
                ptr += 1
                if current_occ[z, y, x] == 0:
                    continue
                if is_simple_point(current_occ, z, y, x):
                    simple_pts.append((z, y, x))
                    current_occ[z, y, x] = 0
            if not simple_pts:
                continue

            endgame_evals += 1
            result = evaluate_occ(current_occ)
            vol_red = 1.0 - result["volume"] / original_volume

            if result["constraints_ok"]:
                total_removed += len(simple_pts)
                endgame_removed += len(simple_pts)
                consecutive_fails = 0
                entry = {
                    "batch": batch_num, "phase": "endgame",
                    "removed": len(simple_pts), "total_removed": total_removed,
                    "volume": result["volume"],
                    "vol_reduction": float(vol_red),
                    "obj":  float(result["obj"]),
                    "vm":   float(result["vm_conservative"]),
                    "comp": float(result["comp_conservative"]),
                }
                history.append(entry)
            else:
                for z, y, x in simple_pts:
                    current_occ[z, y, x] = 1
                consecutive_fails += 1
                if consecutive_fails >= 15:
                    break

        if consecutive_fails >= 15 or endgame_evals >= ENDGAME_MAX_EVALS:
            break
        if eg_round > 0 and endgame_removed == 0:
            break

    phase2_vol = int(current_occ.sum())

    # ═══════════════════════════════════════════════════════════
    # POST-PROCESSING: fill holes, remove shards
    # ═══════════════════════════════════════════════════════════
    # Fill small enclosed air pockets
    air_mask = (current_occ == 0)
    air_labels, n_air_comp = label(air_mask, structure=_STRUCT6)
    boundary_labels = set()
    for face in [air_labels[0,:,:], air_labels[-1,:,:],
                 air_labels[:,0,:], air_labels[:,-1,:],
                 air_labels[:,:,0], air_labels[:,:,-1]]:
        boundary_labels.update(face[face > 0].tolist())
    holes_filled = 0
    for lbl in range(1, n_air_comp + 1):
        if lbl in boundary_labels:
            continue
        comp = air_labels == lbl
        sz = int(comp.sum())
        if sz <= 50:
            current_occ[comp] = 1
            holes_filled += sz

    # Remove shard voxels
    kernel = _STRUCT6.astype(np.float32)
    kernel[1, 1, 1] = 0
    total_spikes = 0
    for i in range(15):
        nc = fftconvolve(current_occ.astype(np.float32), kernel, mode='same')
        nc = np.round(nc).astype(np.int32)
        shards = (current_occ > 0) & (nc < 2)
        n = int(shards.sum())
        if n == 0:
            break
        coords = np.argwhere(shards)
        nc_vals = nc[shards]
        order = np.argsort(nc_vals)
        coords = coords[order]
        removed_pass = 0
        for coord in coords:
            z, y, x = coord
            if current_occ[z, y, x] == 0:
                continue
            if is_simple_point(current_occ, z, y, x):
                current_occ[z, y, x] = 0
                total_spikes += 1
                removed_pass += 1
        if removed_pass == 0:
            break

    # ═══════════════════════════════════════════════════════════
    # FINAL EVALUATION
    # ═══════════════════════════════════════════════════════════
    total_time = time.time() - t_start
    final_vol = int(current_occ.sum())
    vol_red = 1.0 - final_vol / original_volume

    final_result = evaluate_occ(current_occ)

    # Per-part breakdown
    part_breakdown = {}
    for pname, pid in [("exterior_wall", PART_EXTERIOR_WALL),
                       ("interior_wall", PART_INTERIOR_WALL),
                       ("roof", PART_ROOF), ("floor", PART_FLOOR)]:
        orig = int(((fixed_part == pid) & (fixed_occ > 0)).sum())
        cur  = int(((fixed_part == pid) & (current_occ > 0)).sum())
        pct  = cur / orig * 100 if orig > 0 else 100
        part_breakdown[pname] = {
            "original": orig, "optimized": cur, "retained_pct": round(pct, 1)
        }

    # Save results
    np.savez_compressed(out_dir / "optimized_occ.npz", data=current_occ)

    summary = {
        "sample_id": sample_id,
        "success": True,
        "method": "sasto_part_aware",
        "volume_original": original_volume,
        "volume_optimized": final_vol,
        "volume_reduction": float(vol_red),
        "volume_reduction_pct": round(vol_red * 100, 2),
        "phase1_volume": phase1_vol,
        "phase2_volume": phase2_vol,
        "total_removed": total_removed,
        "holes_filled": holes_filled,
        "spikes_removed": total_spikes,
        "constraints_satisfied": final_result["constraints_ok"],
        "vm_conservative": float(final_result["vm_conservative"]),
        "disp_conservative": float(final_result["disp_conservative"]),
        "comp_conservative": float(final_result["comp_conservative"]),
        "comp_limit": float(comp_limit) if comp_limit else None,
        "vm_utilization": float(final_result["vm_conservative"] / MAX_VON_MISES),
        "comp_utilization": float(final_result["comp_conservative"] / comp_limit) if comp_limit else None,
        "pred_mean": final_result["pred_mean"].tolist(),
        "pred_std": final_result["pred_std"].tolist(),
        "total_time_seconds": total_time,
        "n_batches": batch_num,
        "part_breakdown": part_breakdown,
        "baseline_targets": targets_json,
        "history": history,
    }

    with open(out_dir / "optimization_summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    if verbose:
        print(f"  {sample_id}: {vol_red:.1%} reduction "
              f"({original_volume:,} -> {final_vol:,}) "
              f"in {total_time:.0f}s, "
              f"constraints={'OK' if final_result['constraints_ok'] else 'VIOLATED'}")

    # Free memory
    del current_occ, fixed_occ, fixed_part, sensitivity
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary


def main():
    print("=" * 70)
    print("BATCH SASTO OPTIMIZATION — 20 Test Models")
    print("=" * 70)
    print()

    # Paths
    runs_v3 = Path("runs/v3")
    data_dir = Path("data/runs_real_128")
    output_dir = runs_v3 / "batch_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Load ensemble (once)
    config_path = runs_v3 / "config.yaml"
    ensemble_dir = runs_v3 / "ensemble"
    config, norm_dict, models = load_ensemble(config_path, ensemble_dir, device)

    # Run optimization on each sample
    results = []
    failed = []

    for i, sid in enumerate(TEST_SAMPLES):
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(TEST_SAMPLES)}] Optimizing sample {sid}")
        print(f"{'='*60}")

        try:
            result = optimize_sample(
                sample_id=sid,
                data_dir=data_dir,
                output_dir=output_dir,
                models=models,
                norm_dict=norm_dict,
                config=config,
                device=device,
            )
            if result:
                results.append(result)
            else:
                failed.append(sid)
        except Exception as e:
            print(f"  [ERROR] {sid}: {e}")
            traceback.print_exc()
            failed.append(sid)

    # ═══════════════════════════════════════════════════════════
    # AGGREGATE RESULTS
    # ═══════════════════════════════════════════════════════════
    print(f"\n\n{'='*70}")
    print("BATCH RESULTS SUMMARY")
    print(f"{'='*70}\n")

    if not results:
        print("No successful optimizations!")
        return

    vol_reds = [r["volume_reduction_pct"] for r in results]
    times = [r["total_time_seconds"] for r in results]
    vm_utils = [r["vm_utilization"] for r in results]
    comp_utils = [r["comp_utilization"] for r in results if r["comp_utilization"] is not None]
    constraints_ok = sum(1 for r in results if r["constraints_satisfied"])

    # Per-part retention averages
    ext_ret = [r["part_breakdown"]["exterior_wall"]["retained_pct"] for r in results]
    int_ret = [r["part_breakdown"]["interior_wall"]["retained_pct"] for r in results]
    roof_ret = [r["part_breakdown"]["roof"]["retained_pct"] for r in results]
    floor_ret = [r["part_breakdown"]["floor"]["retained_pct"] for r in results]

    aggregate = {
        "n_samples": len(results),
        "n_failed": len(failed),
        "failed_ids": failed,
        "volume_reduction_pct": {
            "mean": round(np.mean(vol_reds), 2),
            "std": round(np.std(vol_reds), 2),
            "min": round(np.min(vol_reds), 2),
            "max": round(np.max(vol_reds), 2),
            "median": round(np.median(vol_reds), 2),
        },
        "runtime_seconds": {
            "mean": round(np.mean(times), 1),
            "std": round(np.std(times), 1),
            "min": round(np.min(times), 1),
            "max": round(np.max(times), 1),
        },
        "constraints_satisfied": constraints_ok,
        "constraints_satisfied_pct": round(constraints_ok / len(results) * 100, 1),
        "vm_utilization": {
            "mean": round(np.mean(vm_utils), 4),
            "std": round(np.std(vm_utils), 4),
        },
        "comp_utilization": {
            "mean": round(np.mean(comp_utils), 4) if comp_utils else None,
            "std": round(np.std(comp_utils), 4) if comp_utils else None,
        },
        "per_part_retention_pct": {
            "exterior_wall": {"mean": round(np.mean(ext_ret), 1), "std": round(np.std(ext_ret), 1)},
            "interior_wall": {"mean": round(np.mean(int_ret), 1), "std": round(np.std(int_ret), 1)},
            "roof": {"mean": round(np.mean(roof_ret), 1), "std": round(np.std(roof_ret), 1)},
            "floor": {"mean": round(np.mean(floor_ret), 1), "std": round(np.std(floor_ret), 1)},
        },
        "per_sample": [
            {
                "sample_id": r["sample_id"],
                "volume_original": r["volume_original"],
                "volume_optimized": r["volume_optimized"],
                "volume_reduction_pct": r["volume_reduction_pct"],
                "constraints_satisfied": r["constraints_satisfied"],
                "runtime_s": round(r["total_time_seconds"], 1),
                "vm_conservative": r["vm_conservative"],
                "comp_conservative": r["comp_conservative"],
            }
            for r in results
        ],
    }

    # Print summary
    print(f"Samples: {len(results)} successful, {len(failed)} failed")
    print(f"Constraints satisfied: {constraints_ok}/{len(results)} "
          f"({aggregate['constraints_satisfied_pct']}%)")
    print(f"\nVolume Reduction (%):")
    print(f"  Mean:   {aggregate['volume_reduction_pct']['mean']:.1f}% "
          f"± {aggregate['volume_reduction_pct']['std']:.1f}%")
    print(f"  Median: {aggregate['volume_reduction_pct']['median']:.1f}%")
    print(f"  Range:  [{aggregate['volume_reduction_pct']['min']:.1f}%, "
          f"{aggregate['volume_reduction_pct']['max']:.1f}%]")
    print(f"\nRuntime: {aggregate['runtime_seconds']['mean']:.0f}s "
          f"± {aggregate['runtime_seconds']['std']:.0f}s")
    print(f"\nPer-Part Retention:")
    for part in ["exterior_wall", "interior_wall", "roof", "floor"]:
        p = aggregate["per_part_retention_pct"][part]
        print(f"  {part:16s}: {p['mean']:.1f}% ± {p['std']:.1f}%")

    print(f"\nPer-Sample Details:")
    print(f"  {'ID':>6s}  {'Vol':>8s}  {'Red%':>6s}  {'OK':>3s}  {'Time':>5s}")
    for s in aggregate["per_sample"]:
        print(f"  {s['sample_id']:>6s}  {s['volume_original']:>8,d}  "
              f"{s['volume_reduction_pct']:>5.1f}%  "
              f"{'Y' if s['constraints_satisfied'] else 'N':>3s}  "
              f"{s['runtime_s']:>5.0f}s")

    # Save aggregate
    with open(output_dir / "aggregate_results.json", "w") as f:
        json.dump(aggregate, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {output_dir}/aggregate_results.json")

    if failed:
        print(f"\nFailed samples: {failed}")


if __name__ == "__main__":
    main()
